import csv
import json
import os
import tempfile

import numpy as np

from popsynth.backgrounds import describe_background
from popsynth.labels import DatasetWriter, export_tables, read_metadata


def test_describe_background_sources():
    g = ["data/backgrounds/inat_hostplants", "data/backgrounds/places365", "data/backgrounds/flowers102"]
    d = describe_background("data/backgrounds/inat_hostplants/acer_297975510.jpg", g)
    assert d == {"background_source": "inat_hostplants", "background_category": "acer"}
    d = describe_background("data/backgrounds/inat_hostplants/glycine max_12.jpg", g)
    assert d["background_category"] == "glycine max"
    d = describe_background("data/backgrounds/places365/r_river/Places365_val_00000026.jpg", g)
    assert d == {"background_source": "places365", "background_category": "river"}
    d = describe_background("data/backgrounds/flowers102/image_00001.jpg", g)
    assert d == {"background_source": "flowers102", "background_category": "flowers102"}


def test_writer_exports_json_and_csv():
    with tempfile.TemporaryDirectory() as root:
        w = DatasetWriter(root, "png")
        img = np.zeros((64, 80, 3), np.float32)
        mask = np.zeros((64, 80), np.float32); mask[10:30, 20:60] = 1
        meta = {"background": "x/acer_1.jpg", "background_source": "inat_hostplants", "background_category": "acer",
                "lighting": "day", "quality_tier": "mild", "effects": ["jpeg_quality"], "jpeg_quality": 70,
                "objects": [{"sprite": "s.png", "sprite_index": 3, "elevation_deg": 30.0, "azimuth_deg": 10.0,
                             "roll_deg": -5.0, "fov_deg": 40.0, "bbox": [20, 10, 60, 30], "visible_frac": 1.0,
                             "object_size_frac": 0.5, "object_px": [40, 20], "mask_area_px": 800}]}
        w.write("img_000000", "train", img, mask, [[20, 10, 60, 30]], meta)
        w.write("img_000001", "val", img, None, [], {"background": "y/image_1.jpg", "background_source": "flowers102",
                                                    "background_category": "flowers102", "lighting": "night",
                                                    "quality_tier": "clean", "effects": [], "objects": []})
        w.close()

        recs = read_metadata(root)
        assert recs[0]["num_objects"] == 1 and recs[1]["num_objects"] == 0
        assert recs[0]["boxes_cxcywh_norm"] == [[0.5, 0.3125, 0.5, 0.3125]]
        o = recs[0]["objects"][0]
        assert o["bbox_xyxy"] == [20, 10, 60, 30] and o["bbox_w_px"] == 40 and o["bbox_area_frac"] == round(800 / 5120, 6)
        assert recs[0]["label_file"] == "labels/train/img_000000.txt" and recs[0]["mask_file"] == "masks/train/img_000000.png"
        assert open(os.path.join(root, recs[0]["label_file"])).read().split()[:3] == ["0", "0.500000", "0.312500"]

        coco = json.load(open(os.path.join(root, "dataset.json")))
        assert len(coco["images"]) == 2 and len(coco["annotations"]) == 1
        a = coco["annotations"][0]
        assert a["bbox"] == [20, 10, 40, 20] and a["category_id"] == 1 and a["area"] == 800
        assert a["metadata"]["elevation_deg"] == 30.0
        assert coco["images"][0]["metadata"]["background_category"] == "acer"
        assert coco["categories"][0]["name"] == "popillia"

        rows = list(csv.DictReader(open(os.path.join(root, "images.csv"))))
        assert [r["class"] for r in rows] == ["popillia", "background"]
        assert rows[0]["num_objects"] == "1" and rows[0]["background_category"] == "acer"
        assert json.loads(rows[0]["boxes_xyxy"]) == [[20, 10, 60, 30]]
        orows = list(csv.DictReader(open(os.path.join(root, "objects.csv"))))
        assert len(orows) == 1 and orows[0]["image_name"] == "img_000000" and orows[0]["roll_deg"] == "-5.0"

        s = json.load(open(os.path.join(root, "summary.json")))
        assert s["num_images"] == 2 and s["num_objects"] == 1
        assert s["per_class"] == {"background": 1, "popillia": 1} and s["lighting"] == {"day": 1, "night": 1}
        assert s["background_source"] == {"flowers102": 1, "inat_hostplants": 1}

        # rebuild from metadata.jsonl gives the same summary
        assert export_tables(root) == s
