function [ uGP ]  = f_uGPLearnParameters( DB ,Nsteps)
% function [ cGP ]  = f_uGPLearnParameters( DB ,Nsteps)
% (c) Markus Froehle, 2019-01-17
%   update: Henk Wymeersch, 2019.02.11
% Description: function learn uGP parameters from training database
% 
% Input: DB is a database struct: 
% DB.y Nx1 vector of measurements, 
% DB.u Nx4 vector of positions 
% DB.NoMeasurements 1x2 # of measurements
% Nsteps is the granularity of the search 
% output: uGP struct with estimated channel parameters

% local parameters
dcvec = linspace( 0.1, 10, Nsteps+1);                               % [m] trial correlation distance
sigma2vec=1:0.2:50;                                                 %  range of values for total variance
dMin = 50;                                                          %  [m] mahalanobis distance type metric to reduce training set based on linearization error
sigma_n = 0.01; % std.dev. measurement noise
sigma2n = sigma_n^2; % measurement noise
Y = DB.y; % measurement vector
N_train = DB.NoMeasurements;


% Step 1: use WLS method to obtain path loss parameters
% -----------------------------------------------------
xTX = DB.mu(:,1); % mean state
yTX = DB.mu(:,2);
xRX = DB.mu(:,3);
yRX = DB.mu(:,4);
d = sqrt( (xTX-xRX).^2 + (yTX-yRX).^2 );
dH =zeros(1,N_train);
for i=1:N_train    
    Sigma = eye(2)*((DB.sigmaTX(i))^2+(DB.sigmaRX(i))^2);    
    zz=[xTX(i)-xRX(i), yTX(i)-yRX(i)];
    dH(i)=zz*inv(Sigma)*zz';
end

F = [ones(N_train,1), -10 * log10( d )];
Jiter = 100; % number of iterations for WLS algorithm
Theta_hatV = zeros(2,Jiter);
% inital guess for parameters:
Theta_hatV(:,1) = [-10,2]' + diag([3, 1]) * randn(2,1);% [L0,eta]'
for j = 1:Jiter
    sigma2_xi = zeros(N_train,1);
    for i = 1:N_train
        J = -10/log(10) * 1/( d(i)^2 ) *...
            [zeros(1,4);[xTX(i)-xRX(i), yTX(i)-yRX(i), -(xTX(i)-xRX(i)), -(yTX(i)-yRX(i))] ];
        Sigma = diag([ DB.sigmaTX(i), DB.sigmaTX(i), DB.sigmaRX(i), DB.sigmaRX(i) ]); % diagonal link position uncertainty
        sigma2_xi(i) = Theta_hatV(:,j)' * J * Sigma*Sigma' * J' * Theta_hatV(:,j);
    end
    W = diag( [sigma2n + sigma2_xi] );
    Winv = W^(-1);
    Theta_hatV(:,j+1) = ( F' * Winv * F  )^(-1) * F' * Winv * Y;
end
Theta_hat = Theta_hatV(:,Jiter);
fprintf('uGP::L0dB=%g, eta=%g\n', Theta_hat(1), Theta_hat(2) );

% Step 2: Remove the mean from the observations
% ----------------------------------------------
Z = Y - F * Theta_hat(:); 

% Step 3: compute total residual variance
% ---------------------------------------
% generate reduced dataset: only use measurements for learning that low
% linearization error
idxMeasurements = dH > dMin;
DBnew.NoMeasurements = sum( idxMeasurements );
DBnew.y = DB.y( idxMeasurements );
DBnew.x = DB.x( idxMeasurements,: );
DBnew.mu = DB.mu( idxMeasurements,: );
DBnew.xhat = DB.xhat( idxMeasurements,: );
DBnew.sigmaTX = DB.sigmaTX( idxMeasurements );
DBnew.sigmaRX = DB.sigmaRX( idxMeasurements );
N_trainnew = DBnew.NoMeasurements; % now based on reduced DB
Znew = Z( idxMeasurements );
sigma2_xinew = sigma2_xi( idxMeasurements ); 

K=length(sigma2vec);
% formulate negative log likelihood
for k=1:K
    LLF(k)=sum(log((sigma2_xinew+sigma2vec(k)))) + sum(Znew.^2./((sigma2_xinew+sigma2vec(k))));
end
[~,bestIndex]=min(LLF);
sigma2tot=sigma2vec(bestIndex) % this should be equal to noise + process + shadowing variances
%alternate expression: sigma2tot = 1/N_trainnew * sum( Znew.^2 )

% Step 4: grid search over dc and sigmaPsi
% -----------------------------------------
sigmaPsivec = linspace( 0, sqrt(sigma2tot-sigma2n ), Nsteps );     % trial standard deviation
logllA = zeros( length(sigmaPsivec),Nsteps+1 );
fractionOfUsedTrainingData=1;                                       % fraction of the training data base used for learning, since this loop is very very slow
for j=1:length(sigmaPsivec)
    fprintf('j:%g\n\r',j);
    for i=1:Nsteps+1
        fprintf('i:%g\n\r',i);
        sigma2psitmp = sigmaPsivec(j)^2;
        sigma2proc = sigma2tot - sigma2psitmp;                        
        logllA(j,i) = f_loglluGP( Znew, dcvec(i), sigmaPsivec(j)^2, sigma2proc , round(fractionOfUsedTrainingData*N_trainnew), DBnew.mu, DBnew.sigmaTX, DBnew.sigmaRX, sigma_n, sigma2_xinew ); % added sigma2_xinew                        
        if logllA(j,i) == -Inf
            logllA(j,i) = Inf;
            fprintf('uGP::logll = -Inf, parameter value excluded \n\r');
        end
    end
end
minMatrix = min(logllA(:));
[row,col] = find(logllA==minMatrix);
dc_estimated = dcvec(col);
sigmaPsi_estimated = sigmaPsivec(row);
sigma2proc_estimated= sigma2tot - sigmaPsi_estimated.^2;
fprintf('uGP::estimated parameter: logLL: %g, sigmaPsi: %g, dc: %g, sigma_proc: %g, sigma_tot:%g \n\r', -(minMatrix), ...
    sigmaPsi_estimated, dc_estimated, sqrt(sigma2proc_estimated), sqrt(sigma2tot) );

%% return estimated parameters
uGP.dc_estimated = dc_estimated;
uGP.sigma2proc_estimated = sigma2proc_estimated;
uGP.sigmaPsi_estimated = sigmaPsi_estimated;
uGP.sigma_n = sigma_n;
uGP.Theta_hat = Theta_hat;
uGP.sigma_tot = sqrt( sigma2tot );
uGP.sigma2_xi = sigma2_xi;

