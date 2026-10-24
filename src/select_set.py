"""Select a date-filtered, fold-diverse set of experimental structures from RCSB PDB."""
import json, random, sys, time
import requests

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
GQL = "https://data.rcsb.org/graphql"
AF2_CUTOFF = "2021-09-30"     # AlphaFold2 training structure cut-off
ESM_CUTOFF = "2020-05-01"     # ESMFold structure-training cut-off (approx.)
MIN_RELEASE = "2022-06-01"    # test structures released after this date
MAX_RELEASE = "2025-06-30"    # newest entries are not yet in the weekly sequence clustering


def _post(url, payload, tries=4):
    for i in range(tries):
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code in (200, 204):
            return r.json() if r.status_code == 200 else {}
        time.sleep(2 * (i + 1))
    r.raise_for_status()


def candidate_entities(rows=3000):
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_accession_info.initial_release_date", "operator": "greater_or_equal", "value": MIN_RELEASE + "T00:00:00Z"}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_accession_info.initial_release_date", "operator": "less_or_equal", "value": MAX_RELEASE + "T00:00:00Z"}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "exptl.method", "operator": "in", "value": ["X-RAY DIFFRACTION"]}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 2.5}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein", "operator": "equals", "value": 1}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count", "operator": "equals", "value": 1}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "entity_poly.rcsb_sample_sequence_length", "operator": "range", "value": {"from": 60, "to": 400, "include_lower": True, "include_upper": True}}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_polymer_entity_annotation.type", "operator": "exists"}} if False else
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count", "operator": "less_or_equal", "value": 400}},
    ]}, "return_type": "polymer_entity", "request_options": {"paginate": {"start": 0, "rows": rows}, "results_verbosity": "compact"}}
    return _post(SEARCH, q)["result_set"]


GQL_Q = """query($ids:[String!]!){polymer_entities(entity_ids:$ids){
 rcsb_id
 entity_poly{pdbx_seq_one_letter_code_can rcsb_sample_sequence_length}
 rcsb_polymer_entity_container_identifiers{reference_sequence_identifiers{database_name database_accession}}
 rcsb_polymer_entity_group_membership{group_id similarity_cutoff aggregation_method}
 entry{rcsb_accession_info{initial_release_date} rcsb_entry_info{resolution_combined}}
}}"""


def fetch_meta(ids):
    out = []
    for i in range(0, len(ids), 50):
        out += _post(GQL, {"query": GQL_Q, "variables": {"ids": ids[i:i + 50]}})["data"]["polymer_entities"]
    return out


def cluster_earliest_release(cluster_id):
    """Earliest release date among all entities in a 30%-identity cluster."""
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_polymer_entity_group_membership.group_id", "operator": "exact_match", "value": cluster_id}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_polymer_entity_group_membership.similarity_cutoff", "operator": "equals", "value": 30}}]},
        "return_type": "polymer_entity", "request_options": {"paginate": {"start": 0, "rows": 1},
        "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "asc"}], "results_verbosity": "compact"}}
    res = _post(SEARCH, q)
    if not res or not res.get("result_set"):
        return None
    eid = res["result_set"][0]
    m = fetch_meta([eid])[0]
    return m["entry"]["rcsb_accession_info"]["initial_release_date"][:10]


def main(n_target=150, n_novel_extra=60, seed=13, out="data/selected.json"):
    random.seed(seed)
    ids = candidate_entities()
    print("candidates:", len(ids), file=sys.stderr)
    random.shuffle(ids)
    metas = fetch_meta(ids[:2000])
    seen_clusters, chosen = set(), []
    for m in metas:
        cl = [c for c in (m["rcsb_polymer_entity_group_membership"] or [])
              if c["aggregation_method"] == "sequence_identity" and c["similarity_cutoff"] == 30]
        if not cl:
            continue
        cid = cl[0]["group_id"]
        if cid in seen_clusters:
            continue
        seq = m["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", "")
        if set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
            continue
        unis = [r["database_accession"] for r in (m["rcsb_polymer_entity_container_identifiers"]["reference_sequence_identifiers"] or []) if r["database_name"] == "UniProt"]
        seen_clusters.add(cid)
        earliest = cluster_earliest_release(cid)
        chosen.append({"entity_id": m["rcsb_id"], "pdb_id": m["rcsb_id"].split("_")[0], "sequence": seq,
                       "uniprot": unis[0] if unis else None, "cluster30": cid,
                       "release_date": m["entry"]["rcsb_accession_info"]["initial_release_date"][:10],
                       "resolution": m["entry"]["rcsb_entry_info"]["resolution_combined"][0],
                       "cluster_earliest_release": earliest,
                       "novel_vs_esmfold": earliest is not None and earliest > ESM_CUTOFF,
                       "novel_vs_af2": earliest is not None and earliest > AF2_CUTOFF})
        # phase 1: first n_target clusters; phase 2: keep adding clusters with no PDB member
        # before the AlphaFold2 cut-off until n_novel_extra such clusters were added
        if len(chosen) > n_target and not chosen[-1]["novel_vs_af2"]:
            chosen.pop()
        if len(chosen) >= n_target and sum(c["novel_vs_af2"] for c in chosen[n_target:]) >= n_novel_extra:
            break
    json.dump(chosen, open(out, "w"), indent=1)
    print(len(chosen), "selected;", sum(c["novel_vs_af2"] for c in chosen), "novel vs AF2 cutoff;",
          sum(c["novel_vs_esmfold"] for c in chosen), "novel vs ESMFold cutoff;",
          sum(c["uniprot"] is not None for c in chosen), "with UniProt")


if __name__ == "__main__":
    main()
