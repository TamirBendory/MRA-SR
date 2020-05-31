function [x_est, post_value, post_dis] = SR_EM(data, noise_level, K, x_init, B, options)

% Inputs:
%       data: contains N observations as columns
%       noise_level: standard deviation of the Gaussian noise
%       K: down-sampling factor (sampling spacing)
%       x_init: initial guess
%       B: if non-empty, the signal is projected onto its lowest B
%       frequcnies at each iteration
%       options.S: the inverse of the prior covariance matrix
%       options.niter: maximal number of EM iterations
%       options.tolerance: stopping criterion for EM iterations
%
% Outpouts:
%       x_est: the estimated signal
%       post_value: log posterior per iteration
%       post_dis: log posterior discrepancy
%
% Tamir Bendory, last updated: 5/31/2020

%% Default values
if  ~isfield(options,'niter') || isempty(options.niter)
    niter = 100; %default value
else
    niter = options.niter;
end
if  ~isfield(options,'tolerance') || isempty(options.tolerance)
    tolerance = 1e-5; %default value
else
    tolerance = options.tolerance;
end
if  ~isfield(options,'verbosity') || isempty(options.verbosity)
    verbosity = 1; %default value
else
    verbosity = options.verbosity;
end
if  ~isfield(options,'S') || isempty(options.S)
    S = []; %default value
else
    S = options.S;
end

%% Preparing for EM iterations
x_est = x_init;
fftdata = fft(data); % Precomputations
sqnormdata = repmat(sum(abs(data).^2, 1), size(data,1), 1);
post_value = zeros(niter, 1);
post_dis = zeros(niter-1,1);

%% EM iterations

for iter = 1 : niter
    [x_new, post_value(iter)] = EM_iteration(x_est, fftdata, sqnormdata, noise_level, K, S, B);
    if iter>2
        post_dis(iter-2) = -(post_value(iter) - post_value(iter-1))/post_value(iter);
    end
    if (mod(iter,10) == 0) && (verbosity == 1) % reporting
        fprintf('iter = %g, posterior discrepancy = %.4g \n', iter, post_dis(iter-2));
    end
    % stopping criterion
    if iter > 2
        if  abs(post_dis(iter-2))<tolerance  && isinf(post_value(iter))==0
            x_est = x_new;
            post_dis = post_dis(1:iter-2);
            post_value = post_value(1:iter);
            fprintf('EM last iteration = %g\n', iter);
            break;
        end
    end
    % update
    x_est = x_new;
    if iter == niter
        fprintf('EM last iteration = %g\n', niter);
    end
end

end

%%  Execute one iteration of EM .
function [x_new, post_value] = EM_iteration(x_est, fftdata, sqnormdata, sigma, K, S, B)

% Inputs:
%       x_est: current estimate
%       fftdata
%       sqnormdata
%       sigma: noise std
%       K: down-sampling factor (sampling spacing)
%       S: the inverse of the prior's covariance
%       B: if non-empty, the signal is projected onto its lowest B
%
% Outpouts:
%       x_new: new signal estimate
%       post_value - log posterior value

M = length(x_est);
N = size(fftdata,2);
T = zeros(M,N);
for i = 1:K
    xk = x_est(i:K:end);
    fftx = fft(xk);
    C = ifft(bsxfun(@times, conj(fftx), fftdata));
    T(i:K:M,:) = -(sqnormdata + norm(xk)^2 - 2*C)/(2*sigma^2);
end
post_value = sum(log(sum(exp(T),1))) - x_est'*S*x_est  ;  % posterior function
T = bsxfun(@minus, T, max(T, [], 1));
W = exp(T);
W = bsxfun(@times, W, 1./sum(W, 1));
a = zeros(M,1);
b = zeros(M,1);
for i = 1:K
    w = W(i:K:M,:);
    b(i:K:end) = ifft(sum(conj(fft(w)).*fftdata, 2));
    a(i:K:end) = sum(w(:));
end
A = diag(a) + S*sigma^2*N;
x_new = A\b;
if ~isempty(B)
    x_new = LP_proj(x_new, B);
end
end

% Execute one iteration of EM .
% function [x_new, LL] = EM_iteration1(x_est, fftdata, sqnormdata, sigma, K, S)
%
% S - the inverse of the prior covariance matrix
% outputs the updated signal and the log likelihood (of the previous
% iteration)
% M = length(x_est);
% N = size(fftdata,2);
% T = zeros(M,N);
% for i = 1:K
%     xk = x_est(i:K:end);
%     fftx = fft(xk);
%     C = ifft(bsxfun(@times, conj(fftx), fftdata));
%     T(i:K:M,:) = -(sqnormdata + norm(xk)^2 - 2*C)/(2*sigma^2);
% end
%
% LL = sum(log(sum(exp(T),1))) - x_est'*S*x_est  ;  % log likelihood function
% T = bsxfun(@minus, T, max(T, [], 1));
% W = exp(T);
% W = bsxfun(@times, W, 1./sum(W, 1));
% a = zeros(M,1);
% b = zeros(M,1);
% for i = 1:K
%     w = W(i:K:M,:);
%     b(i:K:end) = ifft(sum(conj(fft(w)).*fftdata, 2));
%     a(i:K:end) = sum(w(:));
% end
% b = b/sigma^2/N;
% A = diag(a)/sigma^2/N + S;
% x_new = A\b;
% end
%
% for  i = 1:N
%     for k = 1:K
%         C = ifft(conj(fft(x_est(k:K:end))).*fftdata(:,i));
%         T(k:K:M,i) = -(sqnormdata(:,i) + norm(x_est((k:K:end)))^2 - 2*C)/(2*sigma^2);
%     end
% end
