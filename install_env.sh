conda create -n PIPNet python=3.9 -y

source activate PIPNet

conda install cudatoolkit=11.3 -c pytorch -y

pip install -e .

