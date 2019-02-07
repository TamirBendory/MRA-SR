function [x, sigma_f, SIGMA] = generate_signal(beta, L)
% generating a signal of length L with desired power spectrum
% output: x - the generated signal
%         sigma_f - the expected power spectrum, decaying like f^beta
%         Sigma - the covariance matrix of x


% The expected power spectrum of the signal (imposing symmetry in the
% Fourier domain)
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
x = x(:)/sqrt(L); % note: the normalization here is important so that the signal will follow the prior.

end
