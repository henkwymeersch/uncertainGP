function y = f_kerneluGP( u1, Sigma1, u2, Sigma2, sigma2_psi, sigma2_proc, dc, sigma2_xi  )
% (c) Markus Froehle, Date: 2019-01-17
% Description: kernel function for uGP model (with location uncertainty)

if (sum(u1 == u2)==4) && (sum(sum(Sigma1 == Sigma2))==16) % if means and sigmas are the same
    y = sigma2_proc + sigma2_psi + sigma2_xi;
    return;
end
I = eye(2); 
Sigma1TX = Sigma1(1:2,1:2);
Sigma1RX = Sigma1(3:4,3:4);
Sigma2TX = Sigma2(1:2,1:2);
Sigma2RX = Sigma2(3:4,3:4);
GammaTX = I + dc^-2 * (Sigma1TX^2 + Sigma2TX^2);
GammaRX = I + dc^-2 * (Sigma1RX^2 + Sigma2RX^2);
deltaTX = (u1([1,2])-u2([1,2]))';
deltaRX = (u1([3,4])-u2([3,4]))';
detTX = GammaTX(1,1) * GammaTX(2,2); % since it is diagonal
detRX = GammaRX(1,1) * GammaRX(2,2); % since it is diagonal
y = sigma2_psi * detTX^(-1/2) * ...
    exp( -1/dc^2 * deltaTX' * ( GammaTX )^-1 * deltaTX ) * ...
    detRX^(-1/2) * ...
    exp( -1/dc^2 * deltaRX' * ( GammaRX )^-1 * deltaRX );

