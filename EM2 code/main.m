clear;
close all;
clc;

%% Problem setup

% Start the parallel pool
% parallel_nodes = 2;
% if isempty(gcp('nocreate'))
%     parpool(parallel_nodes, 'IdleTimeout', 240);
% end

%seed = rng(475847);

L = 60; % signal length
K = 6; % down-sampling factor 
assert(mod(L,K)==0,'Please choose K to be a divsor of L');

% Generating a signal with decaying power spectrum
beta = 2;

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

sigma_f = sigma_f/norm(sigma_f);

% Ground truth signal
%SIGMA = circulant(sqrt(L)*ifft(sigma_f)); % signal's covariance matrix 
SIGMA = circulant(ifft(sigma_f)); % signal's covariance matrix 

x_true = mvnrnd(zeros(L,1), SIGMA);
x_true = x_true(:);

% Number of measurements
N = 1e7;

% Noise level
snr = 1;
noise_level = norm(x_true)/sqrt(snr*L);

% saving the parameters of the problem
save('parameters');

% Generate the data
data = generate_observations(x_true, N, noise_level, K);

%% EM

x_init = mvnrnd(zeros(L,1), SIGMA);
x_init = x_init(:);
S = inv(SIGMA); % Note: S is the inverse of the covarince matrix
niter = 1000; % maximal number of iteration for EM
tolerance = 1e-8; % tolerance for stopping criterion 

%levels = 2;

[x_est, EM_discrepancy] = SR_EM(data, noise_level, K, x_init, S, niter, tolerance);
%[x_est, EM_discrepancy] = SR_EM_FM(data, noise_level, K, x_init, S, niter, tolerance, levels);

%% Evaluate quality of recovery

%x_est = x_est/norm(x_est)*norm(x_true); % ??
x_est = align_to_reference(x_est, x_true);
err_x = norm(x_est - x_true) / norm(x_true);
fprintf('recovery error = %.4g\n', err_x);

figure; 

subplot(2,1,1);
hold on;
plot(1:L, x_true);
plot(1:L, x_est);
legend('true','estimated')
title(strcat('err = ', num2str( err_x)));
axis tight

subplot(2,1,2);
last_iter = length(EM_discrepancy); 
plot(1:last_iter, EM_discrepancy);
xlabel('iteration');
ylabel('EM discrepancy');
title(strcat('last iteration = ', num2str(last_iter)));
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
axis tight
