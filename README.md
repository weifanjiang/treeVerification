# Robustness Verification of Tree-based Models with feature bound

Based on https://github.com/chenhongge/treeVerification.

This implements section 3.4 of the paper [Robustness Verification of Tree-based Models](https://arxiv.org/pdf/1906.03849.pdf), which uses a Box instead of a Ball for the ensembled tree model verification.

## Run Verification

```
git clone --recurse-submodules git@github.com:weifanjiang/treeVerification.git
cd treeVerification
./compile.sh
./treeVerify config/twitter_spam_baseline.json
```

## Result

### Result for twitter spam dataset

|| T      | L         | Baseline  | Robust |
|---| ------------- |-------------| ----- | ----- |
|l1| 4    | 1 | 5.55112e-15 | 0.0133133 |
|linf| 4 | 1 | 2.22045e-16 | 0.000532532 |
