function [logll] = f_logllcGP(upsilon, dc, sigma2psi, sigma2proc, N_train, z_train, sigma_n, p_train)
% (c) Markus Froehle, Date: 2019-01-17
% Description: function computes log likelihood for a given set of input
% values - cGP case

% build covariance matrix K:
C_train = zeros(N_train,N_train);
for i=1:N_train
    for j=1:N_train
        if i<=j
            % TX->RX:
            z_i = z_train(i,:);
            z_j = z_train(j,:);
            C_train(i,j) = f_kernelcGP( z_i, z_j, sigma2psi, sigma2proc, dc, p_train );            
        end
    end
end
% make C symmetric:
C_train = triu(C_train)+triu(C_train,1)';
K = C_train + sigma_n^2 .* eye(size(C_train));
Kinv = K^-1;

logll =  logdet(K ) + upsilon' * Kinv * upsilon;
