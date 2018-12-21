function [X, shifts] = generate_observations(x, N, sigma, K)

% Given a signal x of length N, generates a matrix X of size L x N such
% that each column of X is a randomly, circularly shifted, L-downsampled version of x with
% i.i.d. Gaussian noise of variance sigma^2 added on top.

x = x(:);
L = length(x);

X = zeros(L, N);
shifts = randi(L, N, 1) - 1;
for n = 1 : N
    X(:, n) = circshift(x, shifts(n));
end

% down-sampling
X = X(1:K:end, :);
% adding Gaussian noise
X = X + sigma*randn(L/K, N);

end