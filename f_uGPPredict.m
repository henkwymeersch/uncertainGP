function [ P_RX_mean, P_RX_var ]  = f_uGPPredict( u_test, Sigma_test, uGPTrainingDB, uGPParam )
% (c) Markus Froehle, 2019-01-17
% Description: function uGP prediction in text locations u_test with uncertainty Sigma_test  from
% database uGPTrainingDB
% output: expected power level and variance

sigma2psi = uGPParam.sigmaPsi_estimated.^2;
sigma2proc = uGPParam.sigma2proc_estimated; 
dc = uGPParam.dc_estimated;
sigma2_xi = uGPTrainingDB.sigma2_xi;
sigma_n = uGPParam.sigma_n;

% build vector k_star
k_star = zeros( uGPTrainingDB.N_train, 1 );
for ii = 1:uGPTrainingDB.N_train
    Sigma_ii = diag([uGPTrainingDB.sigmaTX(ii), uGPTrainingDB.sigmaTX(ii),...
        uGPTrainingDB.sigmaRX(ii),uGPTrainingDB.sigmaRX(ii)]);
    k_star(ii) = f_kerneluGP(u_test,Sigma_test, uGPTrainingDB.u_train(ii,:),...
        Sigma_ii,sigma2psi,sigma2proc,dc, sigma2_xi(ii) );
end

% compute sigma2_xi for u_test:
xTX = u_test(1); % mean state
yTX = u_test(2);
xRX = u_test(3);
yRX = u_test(4);
d = sqrt( (xTX-xRX).^2 + (yTX-yRX).^2 );
J = -10/log(10) * 1/( d^2 ) *...
    [zeros(1,4);[xTX-xRX, yTX-yRX, -(xTX-xRX), -(yTX-yRX)] ];
Sigma = Sigma_test; % diagonal
sigma2_xi_u_test = uGPParam.Theta_hat' * J * Sigma * J' * uGPParam.Theta_hat;
if d == 0
%    fprintf('debug: d==0, set sigma2_xi_u_test=0\n');
    sigma2_xi_u_test = 0;
end

k_starstar = f_kerneluGP( u_test, Sigma_test, u_test, Sigma_test, sigma2psi, sigma2proc, dc , sigma2_xi_u_test);

% predicted mean: % use mean function derivation
% det. PL:
L0dB = uGPParam.Theta_hat(1);
eta = uGPParam.Theta_hat(2);
x = u_test(1);
y = u_test(2);
u = u_test(3);
v = u_test(4);

mu_TX = [x;y];
mu_RX = [u;v];
mu_diff = mu_TX - mu_RX;
sigmaTX = Sigma_test(1,1);
sigmaRX = Sigma_test(3,3);
sigma = sqrt( sigmaTX^2 + sigmaRX^2 );

mu_star = ( mu_diff(1) - 1i * mu_diff(2) ) /( sigma * sqrt(2) );

if mu_star == 0 % if distance is zero
    mu = L0dB;
else
    mu = L0dB - 10*eta * 1/log(10) *  (1/2)* ...
        ( log(abs(mu_star)^2) + expint( abs(mu_star)^2 ) + log( 2*sigma^2 ) );
end

% mean of database measurements:
muX = zeros(uGPTrainingDB.N_train,1);

for i = 1 : uGPTrainingDB.N_train
    x = uGPTrainingDB.u_train(i,1);
    y = uGPTrainingDB.u_train(i,2);
    u = uGPTrainingDB.u_train(i,3);
    v = uGPTrainingDB.u_train(i,4);
    
    mu_TX = [x;y];
    mu_RX = [u;v];
    mu_diff = mu_TX - mu_RX;
    sigmaTX = Sigma_test(1,1);
    sigmaRX = Sigma_test(3,3);
    sigma = sqrt( sigmaTX^2 + sigmaRX^2 );
    
    mu_star = ( mu_diff(1) - 1i * mu_diff(2) ) /( sigma * sqrt(2) );
    
    if mu_star == 0 % if distance is zero
        muX(i) = L0dB;
    else
        muX(i) = L0dB - 10*eta * 1/log(10) *  (1/2)* ...
        ( log(abs(mu_star)^2) + expint( abs(mu_star)^2 ) + log( 2*sigma^2 ) );
    end

end

% Shadowing
P_RX_mean = mu + k_star' * uGPTrainingDB.Kinv * (uGPTrainingDB.y_train - muX ); %

% predicted variance:
P_RX_var  = k_starstar - k_star' * uGPTrainingDB.Kinv * k_star;  % Note: k_starstar = sigma2Proc + simga2Psi + sigma2Xi;



