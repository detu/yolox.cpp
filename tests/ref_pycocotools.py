# Reference numbers from pycocotools (pip install pycocotools):
#   python ref_pycocotools.py gt.json dt.json
import sys, io, contextlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt = COCO(sys.argv[1]); 
with contextlib.redirect_stdout(io.StringIO()):
    dt = gt.loadRes(sys.argv[2]); E = COCOeval(gt, dt, "bbox"); E.evaluate(); E.accumulate()
buf = io.StringIO()
with contextlib.redirect_stdout(buf): E.summarize()
print(buf.getvalue(), end="")
for s in E.stats: print("%.12f" % s)
p = E.eval["precision"]
for k in range(p.shape[2]):
    x = p[:, :, k, 0, -1]; x = x[x > -1]; print("ap %.12f" % (x.mean() if x.size else -1))
