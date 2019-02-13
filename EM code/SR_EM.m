function [x_est, LL, LL_dis] = SR_EM(data, noise_level, K, x_init, S, B, niter, tolerance, verbosity)

% data contains N observations as columns, each of length L/K

% outpout: x_est - the estimated signal
%          LL - the log likelihood per iteration
%          LL_dis - log likelihood discrepancy
%
% noise_level - std of the Gaussian noise
% K - down-sampling factor
% x_init - initial guess
% S - the inverse of the prior covariance matrix
% niter - max EM iterations
% tolerance - stopping criterion for EM iterations

if ~exist('niter', 'var') || isempty(niter)
    niter = 1000;
    fprintf('set default niter = 1000\n');
end

if ~exist('tolerance', 'var') || isempty(tolerance)
    tolerance = 1e-5;
    fprintf('set default tolerance = 1e-5\n');
end

if ~exist('S', 'var') || isempty(S)
    S = 0;
end

x_est = x_init;

% Precomputations on the observations
fftdata = fft(data);
sqnormdata = repmat(sum(abs(data).^2, 1), size(data,1), 1);

LL = zeros(niter, 1);
LL_dis = zeros(niter-1,1);

for iter = 1 : niter
    
    % EM iteration
    [x_new, LL(iter)] = EM_iteration(x_est, fftdata, sqnormdata, noise_level, K, S);
    x_new = LP_proj(x_new, B);
    
    if iter>2
        LL_dis(iter-2) = -(LL(iter) - LL(iter-1))/LL(iter);
    end
    
    if (mod(iter,10) == 0) && (verbosity == 1) 
        fprintf('iter = %g, LL discrepancy = %.4g \n', iter, LL_dis(iter-2));
    end
    
    if iter > 2
        if  LL_dis(iter-2) < tolerance
            x_est = x_new;
            LL_dis = LL_dis(1:iter-2);
            LL = LL(1:iter);
            fprintf('EM last iteration = %g\n', iter);
            break;
        end
    end
    
    x_est = x_new;
    
    if iter == niter
        fprintf('EM last iteration = %g\n', niter);
    end
end

end


% Execute one iteration of EM .
function [x_new, LL] = EM_iteration(x_est, fftdata, sqnormdata, sigma, K, S)

% S - the inverse of the prior covariance matrix
% outputs the updated signal and the log likelihood (of the previous
% iteration)
L = length(x_est);
N = size(fftdata,2);
T = zeros(L,N);
for i = 1:K
    xk = x_est(i:K:end);
    fftx = fft(xk);
    C = ifft(bsxfun(@times, conj(fftx), fftdata));
    T(i:K:L,:) = -(sqnormdata + norm(xk)^2 - 2*C)/(2*sigma^2);
end

LL = sum(log(sum(exp(T),1)));  % log likelihood function
T = bsxfun(@minus, T, max(T, [], 1));
W = exp(T);
W = bsxfun(@times, W, 1./sum(W, 1));

a = zeros(L,1);
b = zeros(L,1);
for i = 1:K
    w = W(i:K:L,:);
    b(i:K:end) = ifft(sum(conj(fft(w)).*fftdata, 2));
    a(i:K:end) = sum(w(:));
end

b = b/sigma^2/N;
A = diag(a)/sigma^2/N + S;
x_new = A\b;

end
