function [ P_RX_mean, P_RX_var ]  = f_cGPPredict( u_test, cGPTrainingDB, cGPParam,p_test_cGP )
% (c) Markus Froehle, 2019-01-17
% Description: function cGP prediction in text locations u_test from
% database cGPTrainingDB
% output: expected power level and variance


sigma2psi = cGPParam.sigmaPsi_estimated.^2;
sigma2proc = cGPParam.sigma2proc_estimated; 
dc = cGPParam.dc_estimated;

% build vector k_star
k_star = zeros( cGPTrainingDB.N_train, 1 );
for ii = 1:cGPTrainingDB.N_train
    k_star(ii) = f_kernelcGP(u_test,cGPTrainingDB.u_train(ii,:),sigma2psi,sigma2proc,dc, p_test_cGP);
end
k_starstar = f_kernelcGP( u_test, u_test, sigma2psi, sigma2proc, dc, p_test_cGP);

% predicted mean:
% det. PL:
L0dB = cGPParam.Theta_hat(1);
eta = cGPParam.Theta_hat(2);
x = u_test(1);
y = u_test(2);
u = u_test(3);
v = u_test(4);
d_TX_RX = sqrt((x-u)^2+(y-v)^2);

if d_TX_RX == 0
    mu = L0dB;
else
    mu = L0dB + -10*eta*log10( d_TX_RX );
end

% mean of database measurements:
xvec = cGPTrainingDB.u_train(:,1);
yvec = cGPTrainingDB.u_train(:,2);
uvec = cGPTrainingDB.u_train(:,3);
vvec = cGPTrainingDB.u_train(:,4);
muX = L0dB + -10*eta*log10(sqrt((xvec-uvec).^2+(yvec-vvec).^2));
muX( isinf( muX ) ) = L0dB; % set mean to L0dB for zero distance!

% Shadowing:
P_RX_mean = mu + k_star' * cGPTrainingDB.Kinv * (cGPTrainingDB.y_train - muX ); 

% predicted variance:
P_RX_var  = ( k_starstar - k_star' * cGPTrainingDB.Kinv * k_star ); 




