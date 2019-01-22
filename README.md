# uncertain GP
Gaussian processes with uncertain input, applied to wireless channel learning and prediction

## Summary
By definition, a Gaussian process (GP) is a collection of random variables, any finite number of which have a joint Gaussian distribution. A GP is defined by an input space *X* and a Gaussian output *y(x)*, where the output has a mean function and a covariance function. Given a collection of input-output pairs, it is then possible to learn the parameters of the mean and covariance function and also to predict the GP at a new input *x*. 

When the input itself is uncertain (e.g., given by a mean and covariance), the learning and prediction steps need to be modified. This code uses classical GPs and so-called uncertain GPs for the task of learning and predicting a wireless communication channel between pairs of communicating agents. 

## Usage

The main file is `maincGPuGP.m` and has a number of parameters that can be set by the user
```
xmax = 30;          % max value in horizontal dimension
ymax = 30;          % max value in vertical dimension
eta = 2;            % pathloss exponent (generally between 1.5 and 4)
dc = 3;             % decorrelation distance (<5 meter indoors and around 50 meter outdoors)
sigmaPsi = 7;       % shadowing std. dev. in dB
L0dB = -10;         % 30 in dB, channel gain (PTX + antenna gain)
p_learning_cGP = 1;     % power of kernel function used for learning cGP (uGP always uses a power of 2)
Nsteps=8;	          % number of grid points for learning
sigmaLow = 1e-9;    % good location std. dev. for training  
sigmaHigh = 10;     % bad location std. dev. for training 
fractionp = 0.7;    % fraction p of bad measurements
NoMeasurements = 1000; % # of measurements used for channel parameter estimation incl. reciprocal
xTX1dim = 15; %horizontal coordinate of TX (does not need to be fixed)
yTX1dim = 15; %vertical coordinate of TX (does not need to be fixed)
```



## Authors
