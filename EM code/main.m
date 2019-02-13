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
% seed = rng(55948);

L = 40; % signal length
K = 4; % down-sampling factor 
assert(mod(L,K)==0,'Please choose K to be a divsor of L');
Nyquist = L/K/2; % Nyquist sampling rate

% number of EM trials 
num_EM_trial = 5;

% Number of measurements
N = 1e4;

% Noise level
snr = 1;

%% Generating a signal with decaying power spectrum 

beta = 1; %decaying rate of the signal's power spectrum
B = 12;  % strict bandwidth (only 2B-1 are non-zero)
[x_true, sigma_f, SIGMA] = generate_signal(beta, L);
x_true = LP_proj(x_true, B); %projecting into B low frequncies

% plotting the true power spectrum compared with the prior
figure; hold on;
plot(1:L, sigma_f);
plot(1:L, abs(fft(x_true)).^2);
plot([Nyquist Nyquist],[0 max(max(sigma_f),max(x_true))],'--')
plot([B B],[0 max(sigma_f)],'--')
legend('prior','true signal','Nyquist','bandlimit');
title('power spectrum of the signal vs. prior');
axis tight
hold off;

%% Generate the data

noise_level = norm(x_true)/sqrt(snr*L); % snr = norm(x)^2/(L*sigma^2)
save('parameters');
data = generate_observations(x_true, N, noise_level, K);

%% EM

S = inv(SIGMA); % Note: S is the inverse of the covarince matrix
niter = 1000; % maximal number of iteration for EM
tolerance = 1e-5; % tolerance for stopping criterion 

% preparing variables for multiple initializations
x_est = zeros(L, num_EM_trial); 
MaxLL = zeros(num_EM_trial, 1);

% plot the Log Likelihood progress
flag_plot_LL = 1;

if flag_plot_LL 
figure; hold on;
end

EM_verbosity = 1;

for iter_em = 1:num_EM_trial
x_init = mvnrnd(zeros(L,1), SIGMA);
x_init = x_init(:);
[x_est(:,iter_em), LL, LL_dis] = SR_EM(data, noise_level, K, x_init, S, B, niter, tolerance, EM_verbosity);
if flag_plot_LL 
plot(LL);
end
MaxLL(iter_em) = LL(end); % maximum of the log-likelihood 
end

if flag_plot_LL 
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
axis tight;
title('Log-likelihood progress');
ylabel('LL');
xlabel('iter');
hold off;
end

% choosing the "best" signal among all trials
[~, ind] = max(MaxLL);

if num_EM_trial>1 
% the variance across initializations provides insight about the difficulty
% of the problem; low variance implies that all trials converged to a
% similar solution
var_em_iter = var(MaxLL);
fprintf('Variance among EM trials = %.4g\n', var_em_iter);
end

x_est_best = x_est(:,ind);
save('x_est_best','x_est_best');

%% Evaluate quality of recovery

x_est_best = align_to_reference(x_est_best, x_true);
err_x = norm(x_est_best - x_true) / norm(x_true);
fprintf('recovery error = %.4g\n', err_x);

figure; 

subplot(2,1,1);
hold on;
plot(1:L, x_true);
plot(1:L, x_est_best);
legend('true','estimated')
title(strcat('err = ', num2str( err_x)));
axis tight
hold off;

err_fft = abs(fft(x_true) - fft(x_est_best))./abs(fft(x_true));
err_fft = err_fft(1:B);
subplot(2,1,2); 
hold on;
plot(1:B, err_fft);
plot([Nyquist Nyquist],[0 max(err_fft)],'--')
title('Relative error as a function of frequency')
axis tight
hold off;