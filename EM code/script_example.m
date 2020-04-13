clear;
close all;
clc;

%% Problem setup

% Start the parallel pool
% parallel_nodes = 2;
% if isempty(gcp('nocreate'))
%     parpool(parallel_nodes, 'IdleTimeout', 240);
% end

B = 10;  % bandwidth
L = 10;  %number of samples
M = L*(2*B+1); % signal length
K = M/L; % down-sampling factor 

% number of EM trials 
num_EM_trial = 1000;

% Number of measurements
N = 100;

% Noise level
snr = 10^40;

%% Generating a signal with the desired power spectrum

%decaying rate of the signal's power spectrum
beta = 1; 
[x_true, sigma_f, SIGMA] = generate_signal(beta, M);
%projecting the signal into B low frequncies
x_true = LP_proj(x_true, B);

%% Generate the data

noise_level = norm(x_true)/sqrt(snr*M); % snr = norm(x)^2/(M*sigma^2)
%save('parameters');
data = generate_observations(x_true, N, noise_level, K);

%% EM

S = inv(SIGMA); % Note: S is the inverse of the covarince matrix
% maximal number of iteration for EM
niter = 10; 
% tolerance for stopping criterion 
tolerance = 1e-10;
% verbosity
EM_verbosity = 1;
% plot the Log Likelihood progress
flag_plot_posterior = 1;

% preparing variables for multiple initializations
x_est = zeros(M, num_EM_trial); 
MaxPosterior = zeros(num_EM_trial, 1);

if flag_plot_posterior 
figure; hold on;
end

for iter_em = 1:num_EM_trial

 if num_EM_trial>1
    fprintf('EM trial #%g out of %g\n',iter_em, num_EM_trial ); 
 end
% initializing EM
x_init = mvnrnd(zeros(M,1), SIGMA);
x_init = x_init(:);
x_init = LP_proj(x_init, B);
[x_est(:,iter_em), post_value, post_dis] = SR_EM1(data, noise_level_em, K, x_init, S, B, niter,...
    tolerance, EM_verbosity);
if flag_plot_posterior 
plot(post_value);
end
% maximal value of the log-likelihood function 
MaxPosterior(iter_em) = post_value(end); 
end

if flag_plot_posterior 
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
axis tight;
title('Log likelihood progress');
ylabel('LL');
xlabel('iter');
hold off;
end

% choosing the "best" signal among all trials
[~, ind] = max(MaxPosterior);

if num_EM_trial>1 
% the variance across initializations provides insight about the difficulty
% of the problem; low variance implies that all trials converged to a
% similar solution.
var_em_iter = var(MaxPosterior);
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
plot(1:M, x_true);
plot(1:M, x_est_best);
legend('true','estimated')
title(strcat('err = ', num2str( err_x)));
axis tight
hold off;

err_fft = abs(fft(x_true) - fft(x_est_best))./abs(fft(x_true));
err_fft = err_fft(1:B);
subplot(2,1,2); 
hold on;
plot(1:B, err_fft);
%plot([Nyquist Nyquist],[0 max(err_fft)],'--')
title('Relative error as a function of frequency')
axis tight
fontsz = 11;
filename = 'script_example.pdf';
%pdf_print_code(gcf, filename, fontsz)