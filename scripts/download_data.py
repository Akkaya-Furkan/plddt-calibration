"""Download experimental mmCIF files and AlphaFold DB predictions for the selected set,
and run ESMFold (ESM Atlas public API) on each sequence. Idempotent: existing files are kept."""
import json, sys, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sel = json.load(open(ROOT / "data/selected.json"))
for d in ("exp", "afdb", "esmfold"):
    (ROOT / "data" / d).mkdir(parents=True, exist_ok=True)


def get(url, tries=4, **kw):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(3 * (i + 1))
    return None


def main():
    log = {"exp_missing": [], "afdb_missing": [], "esm_failed": []}
    for k, e in enumerate(sel):
        eid, pdb = e["entity_id"], e["pdb_id"]
        f = ROOT / f"data/exp/{pdb}.cif"
        if not f.exists():
            r = get(f"https://files.rcsb.org/download/{pdb}.cif")
            if r is None:
                log["exp_missing"].append(eid)
            else:
                f.write_text(r.text)
        if e["uniprot"]:
            f = ROOT / f"data/afdb/{e['uniprot']}.pdb"
            if not f.exists():
                meta = get(f"https://alphafold.ebi.ac.uk/api/prediction/{e['uniprot']}")
                r = get(meta.json()[0]["pdbUrl"]) if meta else None
                if r is None:
                    log["afdb_missing"].append(eid)
                else:
                    f.write_text(r.text)
        f = ROOT / f"data/esmfold/{eid}.pdb"
        if not f.exists():
            for i in range(4):
                r = requests.post("https://api.esmatlas.com/foldSequence/v1/pdb/", data=e["sequence"], timeout=300)
                if r.status_code == 200 and r.text.startswith("HEADER"):
                    f.write_text(r.text)
                    break
                time.sleep(10 * (i + 1))
            else:
                log["esm_failed"].append(eid)
        print(k + 1, len(sel), eid, flush=True)
    json.dump(log, open(ROOT / "data/download_log.json", "w"), indent=1)
    print(log)


if __name__ == "__main__":
    main()
