function [logll] = f_loglluGP(upsilon, dc, sigma2psi, sigma2proc, N_train, u_train, Sigma_train_all_RX, Sigma_train_all_TX, sigma_n, sigma2_xi_all)
% (c) Markus Froehle, Date: 2019-01-17
% Description: function computes log likelihood for a given set of input
% values - uGP case

upsilon=upsilon(1:N_train);
% build covariance matrix K:
C_train = zeros(N_train,N_train);
for i=1:N_train
    for j=1:N_train
        if i<=j
            % TX->RX:
            u_i = u_train(i,:);
            u_j = u_train(j,:);            
            Sigma_train_i = diag([Sigma_train_all_TX(i),Sigma_train_all_TX(i), ...
                Sigma_train_all_RX(i), Sigma_train_all_RX(i) ]);
            Sigma_train_j = diag([Sigma_train_all_TX(j),Sigma_train_all_TX(j), ...
                Sigma_train_all_RX(j), Sigma_train_all_RX(j) ]);
            C_train(i,j) = f_kerneluGP( u_i, Sigma_train_i, u_j, Sigma_train_j, sigma2psi, sigma2proc, dc, sigma2_xi_all(i) );
        end
    end
end
% make C symmetric:
C_train = triu(C_train)+triu(C_train,1)';
K = C_train + sigma_n^2 .* eye(size(C_train)) ;
Kinv = K^(-1);
logll =  logdet( K ) + upsilon' * Kinv * upsilon;