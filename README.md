
ONT (ONT =  ONEAT Training) contains the programs for training neural networks for static (Macrocheate, Mature and Non Mature P1 cells) and dynamic (division and apoptosis) events. The program is Keras and tensorflow based python program which consists of CNN and LSTM based networks for action recognition (division, apoptosis, cell rearrangements can be thought of as action events) and pure CNN networks for cell type recognition. The networks used in this program are fully convolutional hence enjoy the privilege of having the fast implementation of the sliding window operation.


# Installation Instructions
In order to use this program tensorflow version 1.15 and keras library are required. 
1. Download python3.7 version of [anaconda](https://www.anaconda.com/distribution/).
2. After downloading it, open the anaconda terminal and set the proxy settings to enable pip and conda installation via terminal.
3. Set up a virtual enviornment type this command at anaconda prompt: conda create -n tensorflowGPU pip python=3.6 tensorflow-gpu==1.15  
4. Activate the virtual enviornment: source activate tensorflowGPU
5. (Optional) Installation of Cuda and CudNN library versions if you have an NVIDIA GPU for trainng and inference (please check if you have a higher version already installed, then skip this step): conda install cudatoolkit=9.0 cudnn=7.6.4
6. Now install other python packages: pip install keras==2.2.5 csbdeep scikit-image elasticdeform opencv-python tifffile sklearn
7. Installation of jupyter notebooks (default user interface (UI)): conda install jupyter notebook
8. Adding Kernels to the Jupyter notebook: conda install -c conda-forge nb_conda_kernels
Now the installation is complete. To use the program you can close the anaconda terminal and open it again and type the following:
1. cd to the program directory (example path, replace the path with your own): cd /home/Desktop/ONT 
1. conda activate tensorflowGPU
2. jupyter notebook

