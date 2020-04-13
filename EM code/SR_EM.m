function [x_est, post_value, post_dis] = SR_EM(data, noise_level, K, x_init, S, B, niter, tolerance, verbosity)

% The data contains N observations as columns

% outpout: x_est - the estimated signal
%          post_value - the log posterior per iteration
%          post_dis - log posterior discrepancy
%
% noise_level - std of the Gaussian noise
% K - down-sampling factor (sampling spacing)
% x_init - initial guess
% S - the inverse of the prior covariance matrix
% niter - max EM iterations
% tolerance - stopping criterion for EM iterations

x_est = x_init;
% Precomputations on the observations
fftdata = fft(data);
sqnormdata = repmat(sum(abs(data).^2, 1), size(data,1), 1);
post_value = zeros(niter, 1);
post_dis = zeros(niter-1,1);

for iter = 1 : niter
    % EM iteration
    [x_new, post_value(iter)] = EM_iteration(x_est, fftdata, sqnormdata, noise_level, K, S);
    x_new = LP_proj(x_new, B); %projecting onto the signal's bandwidth
    
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
    
    x_est = x_new;
    
    if iter == niter
        fprintf('EM last iteration = %g\n', niter);
    end
end

end


% Execute one iteration of EM .
function [x_new, post_value] = EM_iteration(x_est, fftdata, sqnormdata, sigma, K, S)

% S - the inverse of the prior covariance matrix
% outputs the updated signal and the log likelihood (of the previous
% iteration)
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
%b = b/sigma^2/N;
%A = diag(a)/sigma^2/N + S;
A = diag(a) + S*sigma^2*N;

if cond(A)>10^14
    k=1;
    while cond(A)>10^14
        A = diag(a) + S*(sigma^(2-0.1*k))*N;
        k = k+1;
    end
end
x_new = A\b;
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
