Robustness Verification of Tree-based Models with feature bound

Based on https://github.com/chenhongge/treeVerification.

This implements section 3.4 of 1906.03849, which uses a Box instead of a Ball for the ensembled tree model verification. The configuration file (`example.json`) no longer needs an `eps_init` argument, instead it needs a path to a json file that contains initial dimensions of the Box.

## Instructions to run

```
git clone git@github.com:weifanjiang/treeVerification.git
cd treeVerification
./compile.sh
./treeVerify example.json
```

The above commands run the program with the breast cancer dataset. The initial dimensions of the box is in `breast_cancer_feature_bound.json`.
