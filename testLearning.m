N=10000;
i=1:N;
i=i/100;
sigma=2;
Z=randn(1,N).*sqrt(i)+randn(1,N)*sigma;
var(Z)-mean(i)
sigma_t=0:0.01:10;
K=size(sigma_t,2);
for k=1:K
    LLF(k)=sum(log((i+sigma_t(k)^2))) + sum(Z.^2./((i+sigma_t(k)^2)));
end
plot(sigma_t,LLF)