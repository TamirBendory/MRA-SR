clear;
close all;
clc;
dbstop if error
rng(123);
%% Problem setup

% Start the parallel pool
% parallel_nodes = 2;
% if isempty(gcp('nocreate'))
%     parpool(parallel_nodes, 'IdleTimeout', 240);
% end

% seeds the random number generator
% seed = rng(55948);

B = 5;  % bandwidth
L = 5;  %number of samples
M = 2*L*(2*B+1); % signal length
K = M/L; % down-sampling factor 

% number of EM trials 
num_EM_trial = 10;

% Number of measurements
N = 10000;

% Noise level
snr = 10;

%% Generating a signal with the desired power spectrum

%decaying rate of the signal's power spectrum
beta = 1; 

[x_true, sigma_f, SIGMA] = generate_signal(beta, M);

%projecting the signal into B low frequncies
x_true = LP_proj(x_true, B);

% % plotting the true power spectrum compared with the prior
% figure; hold on;
% plot(1:M, sigma_f);
% plot(1:M, abs(fft(x_true)).^2);
% plot([Nyquist Nyquist],[0 max(max(sigma_f),max(x_true))],'--')
% plot([B B],[0 max(sigma_f)],'--')
% legend('prior','true signal','Nyquist','bandlimit');
% title('power spectrum of the signal vs. prior');
% axis tight
% hold off;

%% Generate the data

noise_level = norm(x_true)/sqrt(snr*M); % snr = norm(x)^2/(L*sigma^2)
save('parameters');
data = generate_observations(x_true, N, noise_level, K);
%% EM

S = inv(SIGMA); % Note: S is the inverse of the covarince matrix
%S = 0;
% maximal number of iteration for EM
niter = 100; 
% tolerance for stopping criterion 
tolerance = 1e-20; %1e-5; 

% preparing variables for multiple initializations
x_est = zeros(M, num_EM_trial); 
MaxLL = zeros(num_EM_trial, 1);

% plot the Log Likelihood progress
flag_plot_LL = 1;

if flag_plot_LL 
figure; hold on;
end

EM_verbosity = 1;

for iter_em = 1:num_EM_trial

 if num_EM_trial>1
    fprintf('EM trial #%g out of %g\n',iter_em,num_EM_trial ); 
 end
 
% initializing EM
x_init = mvnrnd(zeros(M,1), SIGMA);
%x_init = x_true + 0.1*randn(size(x_init'));
x_init = x_init(:);
x_init = LP_proj(x_init, B);

[x_est(:,iter_em), LL, LL_dis] = SR_EM(data, noise_level, K, x_init, S, B, niter,...
    tolerance, EM_verbosity);

if flag_plot_LL 
plot(LL);
end
% maximal value of the log-likelihood function 
MaxLL(iter_em) = LL(end); 
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
% similar solution.
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