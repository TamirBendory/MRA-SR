clear;
close all;
clc;

%% Problem setup

% Start the parallel pool
parallel_nodes = 2;
if isempty(gcp('nocreate'))
    parpool(parallel_nodes, 'IdleTimeout', 240);
end

L = 20; % signal length
K = 2; % down-sampling factor 
assert(mod(L,K)==0,'Please choose K to be a divsor of L');

% Generating a signal with decaying power spectrum
beta = 3;

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

% Ground truth signal
SIGMA = circulant(ifft(sigma_f)); % signal's covariance matrix 
x_true = mvnrnd(zeros(L,1), SIGMA);
x_true = x_true(:);

% Number of measurements
N = 1e4;

% Noise level
snr = 10;
noise_level = sqrt(norm(x_true)^2/L/snr); % std of the Gaussian noise

% saving the parameters of the problem
save('parameters');

% Generate the data
data = generate_observations(x_true, N, noise_level, K);

%% EM

% Initialization: The initial guess is drawn from the prior 
x_init = mvnrnd(zeros(L,1), SIGMA);
x_init = x_init(:);
S = inv(SIGMA); % Note: S is the inverse of the covarince matrix!
niter = 1000; % maximal number of iteration for EM
tolerance = 1e-8; % tolerance for stopping criterion 
[x_est, EM_discrepancy] = SR_EM(data, noise_level, K, x_init, S, niter, tolerance);

% saving results
%clear data
%save('XP_data')

%% Evaluate quality of recovery

x_est = align_to_reference(x_est, x_true);
err_x = norm(x_est - x_true) / norm(x_true);
fprintf('recovery error = %.4g\n', err_x);

figure; 

subplot(2,1,1);
hold on;
plot(1:L,x_true);
plot(1:L,x_est);
legend('true','estimated')
title(strcat('err = ', num2str( err_x)));

% Note that this figure draws the error between consecutive EM iterations, which
% can computed in practice. This is NOT the error with respect to the true
% signal. 
subplot(2,1,2);
last_iter = length(EM_discrepancy); 
plot(1:last_iter, EM_discrepancy);
xlabel('iteration');
ylabel('EM discrepancy');
title(strcat('last iteration = ', num2str(last_iter)));
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
axis tight
