function [x, sigma_f, SIGMA] = generate_signal(beta, L)

% generating a signal of length L with a 1/f^(beta) decaying power spectrum
% output:  x - the generated signal
%         sigma_f - the expected power spectrum, decaying like 1./f^beta
%         Sigma - the covariance matrix of x

sigma_f = zeros(L,1);
sigma_f(1) = 1;
if mod(L,2)==0
    sigma_f(2:L/2+1) = 1./((2:L/2+1).^beta);
    sigma_f(L/2+2:L) = flipud(sigma_f(2:L/2));
else
    sigma_f(2:(L+1)/2) = 1./((2:(L+1)/2).^beta);
    sigma_f((L+1)/2+1:L) = flipud(sigma_f(2:(L+1)/2));
end

SIGMA = circulant(ifft(sigma_f)); % signal's covariance matrix
x = mvnrnd(zeros(L,1), SIGMA);
x = x(:)/sqrt(L); 

end
