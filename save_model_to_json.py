import xgboost as xgb
import pickle
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_svmlight_file
from IPython.display import display, HTML
import sklearn
import sys
import pprint
from collections import deque
import tqdm

"""
This script saves all XGBOOST and SKLEARN models under "ROOT" directory to JSON
"""

ROOT = "models"

# recusion
def get_children(cur_tree, node_id, depth):
    fid = cur_tree.feature[node_id]
    # non leaf
    if fid != -2:
        thisitem = {"nodeid": node_id, "depth": depth, "split": int(fid),
                    "split_condition": cur_tree.threshold[node_id],
                    "yes": int(cur_tree.children_left[node_id]),
                    "no": int(cur_tree.children_right[node_id]),
                    "missing": int(cur_tree.children_left[node_id]),
                    "children": [
                        get_children(cur_tree, int(cur_tree.children_left[node_id]), depth+1),
                        get_children(cur_tree, int(cur_tree.children_right[node_id]), depth+1)
                    ]}
    else:
        # leaf
        output_vec = cur_tree.value[node_id][0]
        malprob = output_vec[1]/(output_vec[0]+output_vec[1])
        thisitem = {"nodeid": node_id, "leaf": malprob}
    return thisitem

def save_sklearn(sklearn_path, json_path):
    rf_cls = pickle.load(open(sklearn_path, 'rb'))

    pretty_forest = []
    # getting the dict of one tree
    for estimator in rf_cls.estimators_:
        cur_tree = estimator.tree_
        root = 0
        depth = 0
        fid = cur_tree.feature[root]
        pretty_tree = {"nodeid": root, "depth": depth, "split": int(fid),
            "split_condition": cur_tree.threshold[root],
            "yes": int(cur_tree.children_left[root]),
            "no": int(cur_tree.children_right[root]),
            "missing": int(cur_tree.children_left[root]),
            "children": [get_children(cur_tree, int(cur_tree.children_left[root]), depth+1),
                        get_children(cur_tree, int(cur_tree.children_right[root]), depth+1)]}
        pretty_forest.append(pretty_tree)
    json.dump(pretty_forest, open(json_path, 'w'), indent=4)

sklearn_models = set()
xgboost_models = set()
for x in os.walk(ROOT, topdown=True):
    if len(x[1]) == 0:
        for fname in x[2]:
            full_name = os.path.join(x[0], fname)
            if full_name.endswith(".bin"):
                xgboost_models.add(full_name)
            if full_name.endswith(".pickle"):
                sklearn_models.add(full_name)

pbar = tqdm.tqdm(total=len(xgboost_models)+len(sklearn_models))
for fname in xgboost_models:
    json_name = fname.replace(".bin", ".json")
    model = xgb.Booster()
    model.load_model(fname)
    model.dump_model(json_name, dump_format='json')
    pbar.update(1)

for fname in sklearn_models:
    json_name = fname.replace(".pickle", ".json")
    save_sklearn(fname, json_name)
    pbar.update(1)
