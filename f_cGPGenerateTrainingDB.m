function [ cGPTrainingDB ]  = f_cGPGenerateTrainingDB( measurementDB, cGPParam,p_train_cGP )
% (c) Markus Froehle, 2019-01-17
% Description: function generate cGP training DB
% Input: measurementDB .. database struct: 
% DB.y Nx1 vector of measurements, 
% DB.u Nx4 vector of postions 
% DB.NoMeasurements 1x2 # of measurements
% cGPParam struct:
% cGP.dc_estimated
% cGP.sigma2proc_estimated
% cGP.sigmaPsi_estimated
%
% output: struct cGPTrainingDB
% cGPTrainingDB.C_train  .. correlation matrix
% cGPTrainingDB.y_train .. training samples including reciprocal link

y_true = measurementDB.y;       % measurements
u_train = measurementDB.xhat;   % noisy estimate of x
N_train = measurementDB.NoMeasurements;

% from cGP learning phase:
sigma2psi = cGPParam.sigmaPsi_estimated.^2;
sigma2proc = cGPParam.sigma2proc_estimated;
dc = cGPParam.dc_estimated;

y_train = y_true; % Note: state u_train is noisy, but observation y_train is taken at correct position z_train

% build covariance matrix K:
C_train = zeros(N_train,N_train);
for i=1:N_train
    for j=1:N_train
        if i<=j
            % TX->RX:
            u_i = u_train(i,:);
            u_j = u_train(j,:);
            C_train(i,j) = f_kernelcGP( u_i, u_j, sigma2psi, sigma2proc, dc, p_train_cGP );
        end
    end
end
% make C symmetric:
C_train = triu(C_train)+triu(C_train,1)';

% build matrix K:
K = C_train + cGPParam.sigma_n^2 .* eye(size(C_train)); % including noise
Kinv = K^(-1);

% return database with training values:
cGPTrainingDB.C_train = C_train;
cGPTrainingDB.y_train = y_train;
cGPTrainingDB.N_train = N_train;
cGPTrainingDB.u_train = u_train;
cGPTrainingDB.Kinv = Kinv;