clear;
close all;
clc;

%% Problem setup

% Start the parallel pool
parallel_nodes = 2;
if isempty(gcp('nocreate'))
    parpool(parallel_nodes, 'IdleTimeout', 240);
end

% seeds the random number generator
seed = rng(5846);

L = 100; % signal length
K = 2; % down-sampling factor 
assert(mod(L,K)==0,'Please choose K to be a divsor of L');

% Generating a signal with decaying power spectrum
beta = 2;
[x_true, sigma_f, SIGMA] = generate_signal(beta, L);

% Number of measurements
N = 1e5;

% Noise level
snr = 1;
noise_level = norm(x_true)/sqrt(snr*L); % snr = norm(x)^2/(L*sigma^2)

% saving the parameters of the problem
save('parameters');

% Generate the data
data = generate_observations(x_true, N, noise_level, K);

%% EM

x_init = mvnrnd(zeros(L,1), SIGMA);
x_init = x_init(:);

S = inv(SIGMA); % Note: S is the inverse of the covarince matrix
niter = 1000; % maximal number of iteration for EM
tolerance = 1e-5; % tolerance for stopping criterion 
[x_est, EM_discrepancy] = SR_EM(data, noise_level, K, x_init, S, niter, tolerance);

% saving estimated signal
save('x_est','x_est');
save('EM_discrepancy','EM_discrepancy');

%% Evaluate quality of recovery

x_est = align_to_reference(x_est, x_true);
err_x = norm(x_est - x_true) / norm(x_true);
fprintf('recovery error = %.4g\n', err_x);

figure; 

subplot(4,1,1);
hold on;
plot(1:L, x_true);
plot(1:L, x_est);
legend('true','estimated')
title(strcat('err = ', num2str( err_x)));
axis tight

err_fft = abs(fft(x_true) - fft(x_est))./abs(fft(x_true));
subplot(4,1,2);
plot(1:L, err_fft);
title('Relative error as a function of frequency')
axis tight

subplot(4,1,3);
hold on;
plot(1:L, abs(fft(x_true)));
plot(1:L, abs(fft(x_est)));
plot(1:L, sqrt(sigma_f));
title('power spectrum')
legend('true', 'estimated', 'prior');
axis tight

subplot(4,1,4);
last_iter = length(EM_discrepancy); 
plot(1:last_iter, EM_discrepancy);
xlabel('iteration');
ylabel('EM discrepancy');
title(strcat('last iteration = ', num2str(last_iter)));
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
axis tight
