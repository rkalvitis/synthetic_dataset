#!/usr/bin/env python
"""Download Creative-Commons photos of Popillia japonica host plants from iNaturalist as backgrounds.

Uses the public iNaturalist API (https://api.inaturalist.org/v1), research-grade observations only,
photos licensed CC0 / CC-BY / CC-BY-NC / CC-BY-SA / CC-BY-NC-SA.  Photos come as the `large`
(1024 px) or `original` (up to 2048 px) size.  Attribution for every file is written to
<out>/attribution.csv (observer, license, observation URL): keep that file with the dataset.

Only observations from June-September are used by default (`--months`), the adult flight season of
Popillia japonica, so backgrounds show summer foliage rather than bare branches or snow.  Note that
plant photos can incidentally contain other insects (useful hard negatives) or, rarely, a Popillia.

Be polite: the script stays well below the API's rate limit (about 1 request per second).

Usage:
  python scripts/download_inat_backgrounds.py [--per-taxon 40] [--size original|large]
                                              [--taxa configs/inat_hostplants.txt] [--out data/backgrounds/inat_hostplants]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.inaturalist.org/v1/observations"
TAXA_API = "https://api.inaturalist.org/v1/taxa"
UA = {"User-Agent": "popsynth-background-downloader/0.1 (thesis research; contact via GitHub)"}
LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"


def get_json(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"  retry ({e})"); time.sleep(5 * (i + 1))
    return {}


def resolve_plant_taxon(name: str) -> int | None:
    """Exact-name lookup restricted to plants, so 'Rosa' or 'Malus' cannot match an animal."""
    q = urllib.parse.urlencode({"q": name, "is_active": "true", "per_page": 10})
    for t in get_json(f"{TAXA_API}?{q}").get("results", []):
        if t.get("name", "").lower() == name.lower() and t.get("iconic_taxon_name") == "Plantae":
            return int(t["id"])
    return None


def read_taxa(path: str) -> list[str]:
    out = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxa", default="configs/inat_hostplants.txt")
    ap.add_argument("--out", default="data/backgrounds/inat_hostplants")
    ap.add_argument("--per-taxon", type=int, default=40, help="photos per taxon")
    ap.add_argument("--size", default="original", choices=["original", "large", "medium"])
    ap.add_argument("--max-per-observation", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between API calls")
    ap.add_argument("--months", default="6,7,8,9",
                    help="observation months to keep (Popillia adult season = foliage season); '' for all")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    attr_path = os.path.join(args.out, "attribution.csv")
    seen = set()
    if os.path.exists(attr_path):
        with open(attr_path) as f:
            seen = {row["file"] for row in csv.DictReader(f)}
    attr = open(attr_path, "a", newline="")
    writer = csv.DictWriter(attr, fieldnames=["file", "taxon", "photo_id", "license", "attribution", "observation_url"])
    if not seen:
        writer.writeheader()

    total = 0
    for taxon in read_taxa(args.taxa):
        got, page = 0, 1
        taxon_id = resolve_plant_taxon(taxon)
        time.sleep(args.sleep)
        if taxon_id is None:
            print(f"[{taxon}] no plant taxon with that exact name, skipped")
            continue
        print(f"[{taxon}] taxon_id {taxon_id}")
        while got < args.per_taxon and page <= 10:
            q = urllib.parse.urlencode({
                "taxon_id": taxon_id, "quality_grade": "research", "photo_license": LICENSES,
                "photos": "true", "per_page": 50, "page": page, "order_by": "votes",
                **({"month": args.months} if args.months else {}),
            })
            data = get_json(f"{API}?{q}")
            time.sleep(args.sleep)
            results = data.get("results", [])
            if not results:
                break
            for obs in results:
                for ph in obs.get("photos", [])[: args.max_per_observation]:
                    if got >= args.per_taxon:
                        break
                    url = ph.get("url", "")
                    lic = ph.get("license_code") or ""
                    if not url or not lic:
                        continue
                    fn = f"{taxon.replace(' ', '_').lower()}_{ph['id']}.jpg"
                    if fn in seen or os.path.exists(os.path.join(args.out, fn)):
                        got += 1
                        continue
                    big = url.replace("square", args.size)
                    try:
                        urllib.request.urlretrieve(big, os.path.join(args.out, fn))
                    except Exception:  # noqa: BLE001
                        try:   # some originals are not served: fall back to large
                            urllib.request.urlretrieve(url.replace("square", "large"), os.path.join(args.out, fn))
                        except Exception as e:  # noqa: BLE001
                            print(f"  skip {ph['id']}: {e}")
                            continue
                    writer.writerow({"file": fn, "taxon": taxon, "photo_id": ph["id"], "license": lic,
                                     "attribution": ph.get("attribution", ""),
                                     "observation_url": f"https://www.inaturalist.org/observations/{obs['id']}"})
                    attr.flush()
                    got += 1; total += 1
                    time.sleep(0.25)
            page += 1
        print(f"  {got} photos")
    attr.close()
    print(f"downloaded {total} new photos into {args.out}; attribution in {attr_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
