# ONT

(ONT)ONeat Training contains the programs for training neural networks for static (Macrocheate, Mature and Non Mature P1 cells) and dynamic (division and apoptosis) events. Incidently ONT is also french for have which can be thought of as if you have this program you can create training datasets and train ONEAT networks. J'ONT ONEAT.

Installation Instructions
In order to use this program tensorflow version 1.15 and keras library are required.

Download python3.7 version of anaconda.
After downloading it, open the anaconda terminal and set the proxy settings to enable pip and conda installation via terminal.
Set up a virtual enviornment type this command at anaconda prompt: conda create -n tensorflowGPU pip python=3.6 tensorflow-gpu==1.15
Activate the virtual enviornment: source activate tensorflowGPU
(Optional) Installation of Cuda and CudNN library versions (please check if you have a higher version already installed, then skip this step): conda install cudatoolkit=9.0 cudnn=7.6.4
Now install other python packages: pip install keras==2.2.5 csbdeep scikit-image elasticdeform opencv-python tifffile sklearn
Installation of jupyter notebooks (default user interface (UI)): conda install jupyter notebook
Adding Kernels to the Jupyter notebook: conda install -c conda-forge nb_conda_kernels Now the installation is complete. To use the program you can close the anaconda terminal and open it again and type the following:
CD to the program directory (example path, replace the path with your own): cd /home/Desktop/ONT
conda activate tensorflowGPU
jupyter notebook

