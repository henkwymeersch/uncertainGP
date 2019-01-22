function [ uGP ]  = f_uGPLearnParameters( DB ,Nsteps)
% function [ cGP ]  = f_uGPLearnParameters( DB ,Nsteps)
% (c) Markus Froehle, 2019-01-17
% Description: function learn uGP parameters from training database
% 
% Input: DB is a database struct: 
% DB.y Nx1 vector of measurements, 
% DB.u Nx4 vector of positions 
% DB.NoMeasurements 1x2 # of measurements
% Nsteps is the granularity of the search 
% output: uGP struct with estimated channel parameters

sigma_n = 0.01; % std.dev. measurement noise
sigma2n = sigma_n^2; % measurement noise
Y = DB.y; % measurement vector
N_train = DB.NoMeasurements;
dcvec = linspace( 0.1, 15, Nsteps);

% use WLS method to obtain Theta_hat --------------------------------------
xTX = DB.mu(:,1); % mean state
yTX = DB.mu(:,2);
xRX = DB.mu(:,3);
yRX = DB.mu(:,4);
d = sqrt( (xTX-xRX).^2 + (yTX-yRX).^2 );
F = [ones(N_train,1), -10 * log10( d )];
Y;
Jiter = 100; % number of iterations for WLS algorithm
Theta_hat = zeros(2,Jiter);
% inital guess for parameters:
Theta_hat(:,1) = [-10,2]' + diag([3, .5]) * randn(2,1);% [L0,eta]'

for j = 1:Jiter
    sigma2_xi = zeros(N_train,1);
    for i = 1:N_train
        J = -10/log(10) * 1/( d(i)^2 ) *...
            [zeros(1,4);[xTX(i)-xRX(i), yTX(i)-yRX(i), -(xTX(i)-xRX(i)), -(yTX(i)-yRX(i))] ];
        Sigma = diag([ DB.sigmaTX(i), DB.sigmaTX(i), DB.sigmaRX(i), DB.sigmaRX(i) ]); % diagonal
        sigma2_xi(i) = Theta_hat(:,j)' * J * Sigma * J' * Theta_hat(:,j);
    end

    W = diag( [sigma2n + sigma2_xi] );
    Winv = W^(-1);

    Theta_hat(:,j+1) = ( F' * Winv * F  )^(-1) * F' * Winv * Y;
end
% estimated parameters L0,eta:
Theta_hat = Theta_hat(:,Jiter);
fprintf('uGP::L0dB=%g, eta=%g\n', Theta_hat(1), Theta_hat(2) );

% total variance to obtain sigma_proc:
% Note: sigma_proc is constant: work on true location x
xTX = DB.x(:,1); % true location
yTX = DB.x(:,2);
xRX = DB.x(:,3);
yRX = DB.x(:,4);
d = sqrt( (xTX-xRX).^2 + (yTX-yRX).^2 );
F = [ones(N_train,1), -10 * log10( d )];
YC_true = Y - F * Theta_hat(:);
sigma2tot_fproc = 1/N_train * sum( YC_true.^2 );


% compute zero mean measurements:
Z = Y - F * Theta_hat(:); % step 3 of WLS in paper p. 7

% generate reduced dataset: only use measurments for learning sigmaTX <0.5
idxMeasurements = DB.sigmaTX < 0.5;
DBnew.NoMeasurements = sum( idxMeasurements );
DBnew.y = DB.y( idxMeasurements );
DBnew.x = DB.x( idxMeasurements,: );
DBnew.mu = DB.mu( idxMeasurements,: );
DBnew.xhat = DB.xhat( idxMeasurements,: );
DBnew.sigmaTX = DB.sigmaTX( idxMeasurements );
DBnew.sigmaRX = DB.sigmaRX( idxMeasurements );
N_trainnew = DBnew.NoMeasurements; % now based on reduced DB
Znew = Z( idxMeasurements );
sigma2_xinew = sigma2_xi( idxMeasurements ); % 2017-02-07 added 

% estimate large scale fading parameters:
% 1. estimate variance: based on reduced set
sigma2tot = 1/N_trainnew * sum( Znew.^2 ); 


% 2. find sigma2proc with cGP method, but p_train =2
sigmaPsivec = linspace( 0, sqrt( sigma2tot_fproc- sigma2n ), Nsteps );
p_learning_uGP = 2;
logll = zeros( Nsteps,Nsteps );
logllmin = inf;
logllmin_idxi = 0;
logllmin_idxj = 0;
for j=1:Nsteps
%     fprintf('j:%g\n\r',j);
    for i=1:Nsteps

        sigma2psitmp = sigmaPsivec(j)^2;
        sigma2proc = sigma2tot_fproc - sigma2psitmp;
        
        if sigma2proc < 0 % set to zero if estimate is negative
            fprintf('uGP::! negative sigma2proc=%g\n\r', sigma2proc);
            sigma2proc = 0;
        end
            logll(j,i) = f_logllcGP( YC_true, dcvec(i), sigmaPsivec(j).^2, sigma2proc, N_train, DB.x, sigma_n, p_learning_uGP );
        if logll(j,i) == -Inf
            logll(j,i) = Inf;
            fprintf('uGP::logll = -Inf, parameter value excluded \n\r');
        end
    end
end

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

% estimated parameters:
sigmaPsi_estimated  = sigmaPsivec(sigmaIdx);
sigma2proc_estimated= sigma2tot_fproc - sigmaPsi_estimated.^2;


% now sigma2proc is estimated based on true location
% estimate other sigma's based on mean location:
% 1. estimate variance: now use based on the mean locations:
sigma2Psi_estimated = sigma2tot - sigma2proc_estimated - sigma2n;

% sanity checks:
if sigma2proc_estimated < 0 % set to zero if estimate is negative
    fprintf('uGP::! negative sigma2proc=%g\n\r', sigma2proc_estimated);
    sigma2proc_estimated = 0;
end
% check if sigma2Psi is positive:
if sigma2Psi_estimated <= 0
    fprintf('uGP::! sigma2Psi_estimated set to 0, sigma2tot=%g, before:sigma2Psi_estimated=%g\n', sigma2tot, sigma2Psi_estimated);
    sigma2Psi_estimated = 0;
end

if ~isreal( sigma2Psi_estimated )
    fprintf 'uGP::#debug: complex!';
    sigma2Psi_estimated = 0;
end

sigmaPsi_estimated = sqrt( sigma2Psi_estimated );



% 3. grid search to find dc:

logll = zeros( Nsteps,1 );
logllmin = inf;
logllmin_idxi = 0;

parfor i=1:Nsteps
    logll(i) = f_loglluGP( Znew, dcvec(i), sigma2Psi_estimated, sigma2proc_estimated, N_trainnew, DBnew.mu, DBnew.sigmaTX, DBnew.sigmaRX, sigma_n, sigma2_xinew ); % added sigma2_xinew
    if logll(i) == -Inf % if some numerical issues appeared:
        logll(i) = Inf;
        fprintf('uGP::logll = -Inf, parameter value excluded \n\r');
    end
end

for i=1:Nsteps
    if real(logll(i)) <= logllmin
        logllmin = logll(i);
        logllmin_idxi = i;
    end
end

% determine minimum of log likelihood:
dcIdx = logllmin_idxi;

% estimated parameters:
dc_estimated = dcvec(dcIdx);


fprintf('uGP::estimated parameter: logLL: %g, sigmaPsi: %g, dc: %g, sigma_proc: %g, sigma_tot:%g \n\r', real(logllmin), ...
    sigmaPsi_estimated, dc_estimated, sqrt(sigma2proc_estimated), sqrt(sigma2tot) );

figure(7);
clf;
plot(dcvec, logll )
title 'uGP::log likelihood'
xlabel 'd_c'
ylabel 'log-ll'
hold on;
h2 = plot(dc_estimated, logllmin, 'g*' );

%% return estimated parameters
uGP.dc_estimated = dc_estimated;
uGP.sigma2proc_estimated = sigma2proc_estimated;
uGP.sigmaPsi_estimated = sigmaPsi_estimated;
uGP.sigma_n = sigma_n;
uGP.Theta_hat = Theta_hat;
uGP.sigma_tot = sqrt( sigma2tot );
uGP.sigma2_xi = sigma2_xi;

