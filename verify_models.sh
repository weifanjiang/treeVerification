./compile.sh
./treeVerify config/breast_cancer_robust_decrease.json > out/breast_cancer_robust_decrease.txt
echo "breast cancer robust decrease done"
./treeVerify config/breast_cancer_robust_increase.json > out/breast_cancer_robust_increase.txt
echo "breast cancer robust increase done"
./treeVerify config/breast_cancer_robust_randhalf.json > out/breast_cancer_robust_randhalf.txt
echo "breast cancer robust randhalf done"
./treeVerify config/higgs_robust_decrease.json > out/higgs_robust_decrease.txt
echo "higgs robust decrease done"
./treeVerify config/higgs_robust_increase.json > out/higgs_robust_increase.txt
echo "higgs robust increase done"
./treeVerify config/higgs_robust_randhalf.json > out/higgs_robust_randhalf.txt
echo "higgs robust randhalf done"
