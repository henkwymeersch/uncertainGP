% (c) Markus Froehle, 2019-01-17
% Description: cGP/uGP hyper-parameter learning + prediction plots
% 
% if you use this code, please cite
% 
% @article{frohle2018channel,
%       title={Channel Prediction With Location Uncertainty for Ad Hoc Networks},
%       author={Fr{\"o}hle, Markus and Charalambous, Themistoklis and Nevat, Ido and Wymeersch, Henk},
%       journal={IEEE Transactions on Signal and Information Processing over Networks},
%       volume={4},
%       number={2},
%       pages={349--361},
%       year={2018}
%}

disp('----------------------------------------------------') 
disp('** cGP/uGP hyper-parameter learning + prediction **')
disp('----------------------------------------------------');

% initialize: 
clearvars; 
close all;
s=RandStream('mt19937ar','Seed',246468);
RandStream.setGlobalStream(s);

% ------------------------------------------------------------
% Part 1: setup the simulation 
% ------------------------------------------------------------
disp('starting the simulation')
xmax = 30;          % max. dimensions
ymax = 30;           
eta = 2;            % pathloss exponent
dc = 3;             % decorrelation distance
sigmaPsi = 7;       % shadowing std. dev. in dB
L0dB = -10;         % 30 in dB, channel gain (PTX + antenna gain)
p_learning_cGP = 1;     % power of kernel function used for learning cGP (uGP always uses a power of 2)
Nsteps=8;	   % number of grid points for learning

% other simulation parameters
sigmaLow = 1e-9;    % good location std. dev. for training  
sigmaHigh = 10;     % bad location std. dev. for training 
fractionp = 0.7;    % fraction p of bad measurements
disp(['the database has a fraction of ' num2str(fractionp ) ' bad measurements']);
NoMeasurements = 1000; % # of measurements used for channel parameter estimation incl. reciprocal
xTX1dim = 15; %location of TX is fixed
yTX1dim = 15;

% ------------------------------------------------------------
% Part 2: generate the channel and measurement database
% ------------------------------------------------------------
disp('generating the channel')
ch = channelWang(eta,dc,sigmaPsi, L0dB);

% generate measurement database 
disp('generating the database')
NoMeasurementsSigmaHigh = floor(NoMeasurements * fractionp); 
NoMeasurementsSigmaLow = NoMeasurements - NoMeasurementsSigmaHigh;

% generate training locations for good database
xTX = ones(NoMeasurementsSigmaHigh/2,1) .* xTX1dim; % fix TX position
yTX = ones(NoMeasurementsSigmaHigh/2,1) .* yTX1dim;
xRX = rand(NoMeasurementsSigmaHigh/2,1) .* xmax; % random
yRX = rand(NoMeasurementsSigmaHigh/2,1) .* ymax;
u_sigmaHigh = [[xTX, yTX, xRX, yRX];[xRX, yRX, xTX, yTX]]; % exact position & reciprocal
        
% generate training locations for bad database
xTX = ones(NoMeasurementsSigmaLow/2,1) .* xTX1dim; % fix TX position
yTX = ones(NoMeasurementsSigmaLow/2,1) .* yTX1dim;
xRX = rand(NoMeasurementsSigmaLow/2,1) .* xmax; % random
yRX = rand(NoMeasurementsSigmaLow/2,1) .* ymax;
u_sigmaLow = [[xTX, yTX, xRX, yRX];[xRX, yRX, xTX, yTX]]; % exact position & reciprocal
        
% generate the two databases (low location uncertainty and high location uncertainty)
if NoMeasurementsSigmaLow ~= 0
    measurementDBSigmaLow = ch.generateNoisyMeasurementDBSigma( NoMeasurementsSigmaLow, sigmaLow, sigmaLow, u_sigmaLow, 1 );
end
if NoMeasurementsSigmaHigh ~= 0
    measurementDBSigmaHigh = ch.generateNoisyMeasurementDBSigma( NoMeasurementsSigmaHigh, sigmaHigh, sigmaHigh, u_sigmaHigh, 1 );
end

% merge the two databases
if NoMeasurementsSigmaLow ~= 0 % not zero
    measurementDB = measurementDBSigmaLow;
    if NoMeasurementsSigmaHigh ~= 0 % append measurements of other set
        measurementDB.y         = [measurementDB.y;measurementDBSigmaHigh.y];
        measurementDB.x         = [measurementDB.x;measurementDBSigmaHigh.x];
        measurementDB.mu        = [measurementDB.mu;measurementDBSigmaHigh.mu];
        measurementDB.xhat      = [measurementDB.xhat;measurementDBSigmaHigh.xhat];
        measurementDB.sigmaTX   = [measurementDB.sigmaTX;measurementDBSigmaHigh.sigmaTX];
        measurementDB.sigmaRX   = [measurementDB.sigmaRX;measurementDBSigmaHigh.sigmaRX];
    end
elseif NoMeasurementsSigmaHigh ~= 0
    measurementDB = measurementDBSigmaHigh;    
end
measurementDB.NoMeasurements = NoMeasurementsSigmaLow + NoMeasurementsSigmaHigh;

% ------------------------------------------------------------
% Part 3: learn the channel parameters and compute GP variables
% ------------------------------------------------------------
disp('perform cGP learning')
cGPParam = f_cGPLearnParameters( measurementDB ,Nsteps,p_learning_cGP);  % cGP learning
cGPTrainingDB = f_cGPGenerateTrainingDB( measurementDB, cGPParam ,p_learning_cGP); % compute correlation matrices etc. 
disp('perform uGP learning')
uGPParam = f_uGPLearnParameters( measurementDB ,Nsteps); % uGP learning
uGPTrainingDB = f_uGPGenerateTrainingDB( measurementDB, uGPParam ); % compute correlation matrices etc. 
                

% ------------------------------------------------------------
% Part 4: prediction in test locations
% ------------------------------------------------------------
disp('perform prediction')
Nx = 40; 
Ny = 40;
mu_cGP = zeros(Ny,Nx);
var_cGP = zeros(Ny,Nx);
mu_uGP = zeros(Ny,Nx);
var_uGP = zeros(Ny,Nx);
ch_true = zeros(Ny,Nx);
% generate test RX locations
xRXvec = linspace( 5, xmax-0.1, Nx )';
yRXvec = linspace( 5, ymax-0.1, Ny )';
SigmaTXPrediction = 1e-9;
SigmaRXPrediction = 1e-9;
SigmaPrediction = blkdiag(SigmaTXPrediction * eye(2), SigmaRXPrediction * eye(2) );
for yidx = 1:Ny
    for xidx = 1:Nx
        z_test = [ xTX1dim, yTX1dim, xRXvec(xidx), yRXvec(yidx) ];
        % cGP:
        [mu_cGP(yidx,xidx), var_cGP(yidx,xidx)]  = f_cGPPredict( z_test,cGPTrainingDB,cGPParam ,p_learning_cGP);
        % uGP:
        [mu_uGP(yidx,xidx), var_uGP(yidx,xidx)]  = f_uGPPredict( z_test, SigmaPrediction, uGPTrainingDB, uGPParam );        
        % true channel
        ch_true(yidx,xidx) = ch.evaluate( xTX1dim, yTX1dim, xRXvec(xidx), yRXvec(yidx) , 1);
    end
end
MSE_cGP = 1/(Nx*Ny) * sum(sum( (mu_cGP - ch_true).^2 ));
MSE_uGP = 1/(Nx*Ny) * sum(sum( (mu_uGP - ch_true).^2 ));

% ------------------------------------------------------------
% Part 5: show results from prediction
% ------------------------------------------------------------
figure(1)
contourf(xRXvec,yRXvec,ch_true)
xlabel('xRX')
ylabel('yRX')
colorbar
title('true channel')

figure(2)
contourf(xRXvec,yRXvec,mu_cGP)
xlabel('xRX')
ylabel('yRX')
colorbar
title(['cGP prediction with MSE = ' num2str(MSE_cGP) ]);

figure(3)
contourf(xRXvec,yRXvec,mu_uGP)
xlabel('xRX')
ylabel('yRX')
colorbar
title(['uGP prediction with MSE = ' num2str(MSE_uGP) ]);

figure(4)
contourf(xRXvec,yRXvec,var_cGP)
xlabel('xRX')
ylabel('yRX')
colorbar
title('cGP variance ');

figure(5)
contourf(xRXvec,yRXvec,var_uGP)
xlabel('xRX')
ylabel('yRX')
colorbar
title('uGP variance ');

disp('done!')
