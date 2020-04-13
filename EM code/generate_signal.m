function [x, sigma_f, SIGMA] = generate_signal(beta, M)

% generating a signal of length L with a 1/f^(beta) decaying power spectrum
% output:  x - the generated signal
%         sigma_f - the expected power spectrum, decaying like 1./f^beta
%         Sigma - the expected covariance matrix of x

sigma_f = zeros(M,1);
sigma_f(1) = 1;
if mod(M,2)==0
    sigma_f(2:M/2+1) = 1./((2:M/2+1).^beta);
    sigma_f(M/2+2:M) = flipud(sigma_f(2:M/2));
else
    sigma_f(2:(M+1)/2) = 1./((2:(M+1)/2).^beta);
    sigma_f((M+1)/2+1:M) = flipud(sigma_f(2:(M+1)/2));
end

SIGMA = circulant(ifft(sigma_f)); % signal's covariance matrix
x = mvnrnd(zeros(M,1), SIGMA);
x = x(:)/norm(x);

end
