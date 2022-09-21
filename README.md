# MA_LearnedOperatorCorrectionISTA
The Parts of the code I used for the results of my Master's thesis "Learned operator correction for the proximal gradient"

## necessary Libraries

the following packages can be installed with pip

+ scipy
+ numpy
+ matplotlib
+ torchinfo
+ dival
+ tqdm

I recommend anaconda for
+ pytorch (https://pytorch.org/get-started/locally/)
+ astra-toolbox (https://github.com/astra-toolbox/astra-toolbox)

and the following libary need to be installed directly
+ odl vs 1.0.0.dev (https://odlgroup.github.io/odl/getting_started/installing_source.html#installing-odl-source)
# content 
+ one can create a schift dataset using 'create_shift_dataset'
+ the jupyter-notebook 'Training' was used to setup a config file with 'conf' and start the training using 'Setup_and_training' (not recommended to do that on a gpu)
+ the jupyter-notebook 'Pictures for master thesis' was used to create the plots in the numerical part of the thesis (can take long on just a Cpu)