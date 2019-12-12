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

The average box for 500 samples on baseline model:

```
{
   0: [5.55112e-17, 2.22045e-16]
   1: [5.55112e-17, 2.22045e-16]
   2: [1.11022e-16, 2.22045e-16]
   3: [1.11022e-16, 2.22045e-16]
   4: [1.11022e-16, 2.22045e-16]
   5: [1.11022e-16, 2.22045e-16]
   6: [0, 2.22045e-16]
   7: [5.55112e-17, 2.22045e-16]
   8: [0, 2.22045e-16]
   9: [5.55112e-17, 2.22045e-16]
   10: [5.55112e-17, 2.22045e-16]
   11: [1.11022e-16, 2.22045e-16]
   12: [1.11022e-16, 2.22045e-16]
   13: [1.11022e-16, 2.22045e-16]
   14: [1.11022e-16, 2.22045e-16]
   15: [1.11022e-16, 2.22045e-16]
   16: [1.11022e-16, 2.22045e-16]
   17: [1.11022e-16, 2.22045e-16]
   18: [1.11022e-16, 2.22045e-16]
   19: [1.11022e-16, 2.22045e-16]
   20: [2.22045e-16, 0]
   21: [2.22045e-16, 5.55112e-17]
   22: [2.22045e-16, 1.11022e-16]
   23: [2.22045e-16, 1.11022e-16]
   24: [2.22045e-16, 1.11022e-16]
}
```

The average box for robust model:

```
{
   0: [4.4632e-05, 0.000178528]
   1: [4.4632e-05, 0.000178528]
   2: [8.92639e-05, 0.000178528]
   3: [8.92639e-05, 0.000178528]
   4: [8.92639e-05, 0.000178528]
   5: [8.92639e-05, 0.000178528]
   6: [0, 0.000178528]
   7: [4.4632e-05, 0.000178528]
   8: [0, 0.000178528]
   9: [4.4632e-05, 0.000178528]
   10: [4.4632e-05, 0.000178528]
   11: [8.92639e-05, 0.000178528]
   12: [8.92639e-05, 0.000178528]
   13: [8.92639e-05, 0.000178528]
   14: [8.92639e-05, 0.000178528]
   15: [8.92639e-05, 0.000178528]
   16: [8.92639e-05, 0.000178528]
   17: [8.92639e-05, 0.000178528]
   18: [8.92639e-05, 0.000178528]
   19: [8.92639e-05, 0.000178528]
   20: [0.000178528, 0]
   21: [0.000178528, 4.4632e-05]
   22: [0.000178528, 8.92639e-05]
   23: [0.000178528, 8.92639e-05]
   24: [0.000178528, 8.92639e-05]
}
```
Under the assumption that each feature has a different cost to change, the robust model has better robust verifiability than the baseline model.
