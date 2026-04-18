echo This script runs all the tests in the "tests-pipeann" directory,
echo and output the results to the "data" directory.
mkdir data

bash $(dirname $0)/tests-pipeann/fig11.sh
bash $(dirname $0)/tests-pipeann/fig12.sh
bash $(dirname $0)/tests-pipeann/fig13.sh
bash $(dirname $0)/tests-pipeann/fig14.sh
bash $(dirname $0)/tests-pipeann/fig15.sh
bash $(dirname $0)/tests-pipeann/fig16.sh
bash $(dirname $0)/tests-pipeann/fig17.sh
bash $(dirname $0)/tests-pipeann/fig18.sh
