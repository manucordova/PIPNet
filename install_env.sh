conda create -n PIPNet python=3.9 -y

conda instal -n PIPNet -c conda-forge numpy scipy -y

conda install -n PIPNet pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
