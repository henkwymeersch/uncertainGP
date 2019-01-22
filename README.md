# Uncertain GP
Gaussian processes with uncertain input, applied to wireless channel learning and prediction

## Summary
By definition, a Gaussian process (GP) is a collection of random variables, any finite number of which have a joint Gaussian distribution. A GP is defined by an input space *X* and a Gaussian output *y(x)*, where the output has a mean function and a covariance function. Given a collection of input-output pairs, it is then possible to learn the parameters of the mean and covariance function and also to predict the GP at a new input *x*. 

When the input itself is uncertain (e.g., given by a mean and covariance), the learning and prediction steps need to be modified. This code uses classical GP (cGP) and so-called uncertain GP (uGP) for the task of learning and predicting a wireless communication channel between pairs of communicating agents. 

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
The code will then go through a number of steps, for both cGP and uGP:
1. Creating the channel model
2. Generating a measurement database with a fraction `fractiop` measurements with high input uncertainty
3. Perform parameter learning
4. Compute data structures for prediction
5. Perform prediction
6. Visualize results

An example result is shown below. 

![example](example.jpeg=250x250)

## Authors
The code was developed by Dr. Markus Fröhle, while he was a PhD student at Chalmers University of Technology. The code is based on the paper 

Fröhle, Markus, Themistoklis Charalambous, Ido Nevat, and Henk Wymeersch. "Channel Prediction With Location Uncertainty for Ad Hoc Networks." *IEEE Transactions on Signal and Information Processing over Networks*, vol. 4, no. 2 (2018): 349-361.

If you plan to use or modify this code, please cite our work:

```
 @article{frohle2018channel,
       title={Channel Prediction With Location Uncertainty for Ad Hoc Networks},
       author={Fr{\"o}hle, Markus and Charalambous, Themistoklis and Nevat, Ido and Wymeersch, Henk},
       journal={IEEE Transactions on Signal and Information Processing over Networks},
       volume={4},
       number={2},
       pages={349--361},
       year={2018}
}
```
