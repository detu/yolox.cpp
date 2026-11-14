# Writes gt<seed>.json / dt<seed>.json to compare test_coco_eval with pycocotools:
#   python gen_random_coco.py 1
import json, random, sys
seed = int(sys.argv[1]); random.seed(seed)
cats = [{"id": 1, "name": "a"}, {"id": 3, "name": "b"}, {"id": 7, "name": "c"}]
images, anns, dets = [], [], []
aid = 1
for i in range(40):
    W, H = random.randint(200, 900), random.randint(200, 900)
    images.append({"id": 1000 + i, "width": W, "height": H, "file_name": f"{i}.jpg"})
    for _ in range(random.randint(0, 12)):
        w = random.choice([random.uniform(3, 30), random.uniform(30, 100), random.uniform(100, 400)])
        h = w * random.uniform(0.5, 2)
        x, y = random.uniform(0, W - 10), random.uniform(0, H - 10)
        c = random.choice([1, 3, 7])
        crowd = 1 if random.random() < 0.08 else 0
        area = w * h * random.uniform(0.6, 1.0)
        anns.append({"id": aid, "image_id": 1000 + i, "category_id": c, "bbox": [x, y, w, h],
                     "area": area, "iscrowd": crowd, "segmentation": [[x, y, x + w, y, x + w, y + h]]})
        aid += 1
        for _ in range(random.randint(0, 3)):
            j = lambda v, s: v + random.gauss(0, s)
            sc = round(random.random(), 2)  # coarse scores -> many ties
            dets.append({"image_id": 1000 + i, "category_id": c if random.random() < 0.85 else random.choice([1, 3, 7]),
                         "bbox": [j(x, w * 0.1), j(y, h * 0.1), abs(j(w, w * 0.1)) + 1, abs(j(h, h * 0.1)) + 1], "score": sc})
    nfp = 130 if i % 9 == 0 else random.randint(0, 8)   # some images exceed maxDets=100
    for _ in range(nfp):
        w, h = random.uniform(2, 200), random.uniform(2, 200)
        dets.append({"image_id": 1000 + i, "category_id": random.choice([1, 3, 7]),
                     "bbox": [random.uniform(0, W), random.uniform(0, H), w, h], "score": round(random.random(), 2)})
json.dump({"images": images, "annotations": anns, "categories": cats, "info": {"x": "é\\n"}}, open(f"gt{seed}.json", "w"))
json.dump(dets, open(f"dt{seed}.json", "w"))
