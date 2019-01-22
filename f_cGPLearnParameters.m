function [ cGP ]  = f_cGPLearnParameters( DB ,Nsteps,p_learning_cGP)
% function [ cGP ]  = f_cGPLearnParameters( DB ,Nsteps)
% (c) Markus Froehle, 2019-01-17
% Description: function learn cGP parameters from training database
% 
% Input: DB is a database struct: 
% DB.y Nx1 vector of measurements, 
% DB.u Nx4 vector of positions 
% DB.NoMeasurements 1x2 # of measurements
% Nsteps is the granularity of the search 
% output: cGP struct with estimated channel parameters

sigma_n = 0.01;         % std.dev. measurement noise
sigma2n = sigma_n^2;    % measurement noise

dcvec = linspace( 0.1, 15, Nsteps);


% least squares estimation of path loss parameters (L0dB and eta)
% using the method from 
% Malmirchegini and Y. Mostofi, ?On the spatial predictability of communication channels,? IEEE Trans. Wirel. Commun., vol. 11, no. 3, pp. 964?978, Mar. 2012.

% model: Y = G*Theta + wSH + wMP; we want to estimate theta
xTX = DB.xhat(:,1);         % assign noise estimate xhat of true x
yTX = DB.xhat(:,2);
xRX = DB.xhat(:,3);
yRX = DB.xhat(:,4);
Y = DB.y;                   % measurement vector
N_train = DB.NoMeasurements;
F = 10 .* log10( sqrt(  (xTX - xRX).^2 + (yTX - yRX).^2 ) );
G = [ones(length(F),1) , -F];
Theta_hat = (G' * G)^(-1) * G' * Y; % contains desired estimates (L0dB and eta)
YG = Y - G * Theta_hat;     % centered version of measurement vector:
fprintf('cGP::L0dB=%g, eta=%g\n', Theta_hat(1), Theta_hat(2) );

% estimate large scale fading parameters:
% 1. estimate variance:
sigma2tot = 1/N_train .* sum( YG.^2 );      % so sigma2proc := sigma2tot - sigma2n - sigma2psi;
sigmaPsivec = linspace( 0, sqrt(sigma2tot - sigma2n), Nsteps );     % now try different sigma2psi
% 2. grid search to find:
logll = zeros( Nsteps,Nsteps );
logllmin = inf;
logllmin_idxi = 0;
logllmin_idxj = 0;
for j=1:Nsteps           % search over all possible sigma2psi
    for i=1:Nsteps          % search over all possible dc
        sigma2psitmp = sigmaPsivec(j)^2;
        sigma2proc = sigma2tot - sigma2psitmp;
        if sigma2proc < 0 % set to zero if estimate is negative            
            sigma2proc = 0;
        end
            logll(j,i) = f_logllcGP( YG, dcvec(i), sigmaPsivec(j).^2, sigma2proc, N_train, DB.xhat, sigma_n, p_learning_cGP );
        if logll(j,i) == -Inf
            logll(j,i) = Inf;           
        end
    end
end

% 3. find best value
for j=1:Nsteps
    for i=1:Nsteps
        if (logll(j,i)) <= logllmin 
            logllmin = logll(j,i);
            logllmin_idxi = i;
            logllmin_idxj = j;
        end
    end
end


% determine minimum of log likelihood:
sigmaIdx = logllmin_idxj;
dcIdx = logllmin_idxi;

% estimated parameters:
sigmaPsi_estimated  = sigmaPsivec(sigmaIdx);
dc_estimated        = dcvec(dcIdx);
sigma2proc_estimated= sigma2tot - sigma2n - sigmaPsi_estimated.^2;
if sigma2proc_estimated < 0 % set to zero if estimate is negative    
    sigma2proc_estimated = 0;
end

fprintf('cGP::estimated parameter: logLL: %g, sigmaPsi: %g, dc: %g, sigma_proc: %g \n\r', real(logllmin), ...
    sigmaPsi_estimated, dc_estimated, sqrt(sigma2proc_estimated) );
figure(6);
clf;
surfc(dcvec, sigmaPsivec, real(logll) )
title 'cGP::log likelihood'
ylabel '\sigma_\Psi'
xlabel 'd_c'
hold on;
plot3(dc_estimated, sigmaPsi_estimated, real(logllmin), 'g*' );


% estimated parameters:
sigma2psi = sigmaPsi_estimated.^2;
dc = dc_estimated;
sigma2proc = sigma2proc_estimated;
cGP.dc_estimated = dc_estimated;
cGP.sigma2proc_estimated = sigma2proc_estimated;
cGP.sigmaPsi_estimated = sigmaPsi_estimated;
cGP.sigma_n = sigma_n;
cGP.Theta_hat = Theta_hat;

