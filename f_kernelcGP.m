function y = f_kernelcGP( x1, x2, sigma2_psi, sigma2_proc, dc, p  )
% (c) Markus Froehle, Date: 2019-01-17
% Description: kernel function for GP model

if x1 == x2
    y = sigma2_proc + sigma2_psi;
    return;
end
y = sigma2_psi * ( exp( -( ( norm(x1([1,2])-x2([1,2])) )^p + ( norm(x1([3,4])-x2([3,4])) )^p )/dc^p ) );
