function [ uGPTrainingDB ]  = f_uGPGenerateTrainingDB( measurementDB, uGPParam )
% (c) Markus Froehle, 2019-01-17
% Description: function generate uGP training DB
% Input: measurementDB .. database struct: 
% DB.y Nx1 vector of measurements, 
% DB.u Nx4 vector of postions 
% DB.NoMeasurements 1x2 # of measurements
% uGPParam struct:
% uGP.dc_estimated
% uGP.sigma2proc_estimated
% uGP.sigmaPsi_estimated
% 
% output: struct uGPTrainingDB
% uGPTrainingDB.C_train  .. correlation matrix
% uGPTrainingDB.y_train .. training samples including reciprocal link


% from channel measurements
y_true = measurementDB.y;
u_train = measurementDB.mu; % mean estimate of true state x
N_train = measurementDB.NoMeasurements;

% from uGP learning phase:
sigma2psi = uGPParam.sigmaPsi_estimated.^2;
sigma2proc = uGPParam.sigma2proc_estimated;
dc = uGPParam.dc_estimated;
sigma2_xi = uGPParam.sigma2_xi;
y_train = y_true; % Note: state u_train is noisy, but observation y_train is taken at correct position z_train

% build covariance matrix K:
C_train = zeros(N_train,N_train);
parfor i=1:N_train
    for j=1:N_train
        if i<=j
            % TX->RX:
            u_i = u_train(i,:);
            u_j = u_train(j,:);
            
            Sigma_train_i = diag([measurementDB.sigmaTX(i),measurementDB.sigmaTX(i), ...
                measurementDB.sigmaRX(i), measurementDB.sigmaRX(i) ]);
            Sigma_train_j = diag([measurementDB.sigmaTX(j),measurementDB.sigmaTX(j), ...
                measurementDB.sigmaRX(j), measurementDB.sigmaRX(j) ]);            
            C_train(i,j) = f_kerneluGP( u_i, Sigma_train_i, u_j, Sigma_train_j, sigma2psi, sigma2proc, dc, sigma2_xi(i) );
        end
    end
end
% make C symmetric:
C_train = triu(C_train)+triu(C_train,1)';

% build matrix K:
K = C_train + uGPParam.sigma_n^2 .* eye(size(C_train)); % including noise
Kinv = K^(-1);

% return database with training values:
uGPTrainingDB.C_train = C_train;
uGPTrainingDB.y_train = y_train;
uGPTrainingDB.N_train = N_train;
uGPTrainingDB.u_train = u_train;
uGPTrainingDB.Kinv = Kinv;
uGPTrainingDB.sigmaTX = measurementDB.sigmaTX;
uGPTrainingDB.sigmaRX = measurementDB.sigmaRX;
uGPTrainingDB.sigma2_xi = sigma2_xi;