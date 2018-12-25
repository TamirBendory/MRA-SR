function [x_est, EM_discrepancy] = SR_EM(data, noise_level, K, x_init, S, niter, tolerance)

% data contains N observations as columns, each of length L/K
% noise_level - std of the Gaussian noise
% K - down-sampling factor
% x_init - initial guess
% S - the inverse of the prior covariance matrix
% niter - max EM iterations
% tolerance - stopping criterion for EM iterations

if nargin == 5
    niter = 1000;
    fprintf('set default niter = 1000\n');
    tolerance = 1e-5;
    fprintf('set default tolerance = 1e-5\n');
end

if nargin == 6
    if niter>1
        tolerance = 1e-5;
        fprintf('set default tolerance = 1e-5\n');
    else
        tolerance = niter;
        niter = 1000;
        fprintf('set default niter = 1000\n');
    end
end

% Initialize
x_est = x_init;

% Precomputations on the observations
fftdata = fft(data);
sqnormdata = repmat(sum(abs(data).^2, 1), size(data,1), 1);

EM_discrepancy = zeros(niter,1);
for iter = 1 : niter

    % EM iteration
    x_new = EM_iteration(x_est, fftdata, sqnormdata, noise_level, K, S);
    % error between consecutive iterations; not the error with respect to
    % the signal itself (which is unknown)
    EM_discrepancy(iter) = norm(align_to_reference(x_new,x_est) - x_est)/norm(x_est);
    
    if mod(iter,100) == 0 
        fprintf('iter = %g, discrepancy = %.4g \n', iter, EM_discrepancy(iter));
        save('XP_data', '-regexp', '^(?!(data)$).') %saving all variables but data
    end
        
    % stopping criterion
    if  EM_discrepancy(iter) < tolerance
        x_est = x_new;
        EM_discrepancy = EM_discrepancy(1:iter);
        fprintf('EM last iteration = %g\n', iter);
        break;
    end
    % update
    x_est = x_new;
    
    if iter == niter
        fprintf('EM last iteration = %g\n', niter);
    end
end

end


% Execute one iteration of EM .
function x_new = EM_iteration(x_est, fftdata, sqnormdata, sigma, K, S)

% S - the inverse of the prior covariance matrix

L = length(x_est);
N = size(fftdata,2);

% wrong code
% W = zeros(L,N);
% 
% for i = 1:K
%     The idea here is to compute the weights for each sub-signal
%     separately, and then combine them. The weights of each sub-signal can be
%     computed efficiently using fft.
%     xk = x_est(i:K:end);
%     fftx = fft(xk);
%     C = ifft(bsxfun(@times, conj(fftx), fftdata));
%     T = -(sqnormdata + norm(xk)^2 - 2*C)/(2*sigma^2);
%     T = bsxfun(@minus, T, max(T, [], 1)); % subtracting the max value
%     W(i:K:L,:) = exp(T);
% end
% 

T = zeros(L,N);
for i = 1:K
    % The idea here is to compute the weights for each sub-signal
    % separately, and then combine them. The weights of each sub-signal can be
    % computed efficiently using fft.
    xk = x_est(i:K:end);
    fftx = fft(xk);
    C = ifft(bsxfun(@times, conj(fftx), fftdata));
    T(i:K:L,:) = -(sqnormdata + norm(xk)^2 - 2*C)/(2*sigma^2);
end

T = bsxfun(@minus, T, max(T, [], 1)); % subtracting the max value
W = exp(T);
W = bsxfun(@times, W, 1./sum(W, 1));

% Y = zeros(L,N);
% Y(1:K:L,:) = ifft(fftdata);
% b = ifft(sum(conj(fft(W)).*fft(Y), 2))/sigma^2;
a = zeros(L,1);
b = zeros(L,1);
for i = 1:K
w = W(i:K:L,:);
b(i:K:end) = ifft(sum(conj(fft(w)).*fftdata, 2))/sigma^2;
a(i:K:end) = sum(w(:)); 
end

A = diag(a)/sigma^2 + S;

% Approximating the accurate A (see notes)
%A = N/sigma^2/K*eye(L) + S;

x_new = A\b;

end
