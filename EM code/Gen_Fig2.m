clear;
close all;
clc;

% seeds the random number generator
seed = rng(9843);
% Start the parallel pool
% parallel_nodes = 2;
% if isempty(gcp('nocreate'))
%     parpool(parallel_nodes, 'IdleTimeout', 240);
% end

%% Problem setup

B = 10;  % strict bandwidth (only 2B+1 are non-zero)
M = 10*(2*B+1); % signal length
%decaying rate of the signal's power spectrum
beta = 0; 
[x_LP, sigma_f, SIGMA] = generate_signal(beta, M);
x_LP = LP_proj(x_LP, B/2); %bandwidth of B=5 
x_true = x_LP;
shift = 10;
x_true(round(M/2)-shift) = x_true(round(M/2)-shift) + 0.5;
x_true(round(M/2)+shift) = x_true(round(M/2)+shift) + 0.5;
x_true = LP_proj(x_true, B);

% Noise level
snr = 1;
noise_level = norm(x_true)/sqrt(snr*M); % snr = norm(x)^2/(L*sigma^2)
% number of EM trials 
num_EM_trial = 1;
% Number of measurements
N = 1e4;

%% Generating data

L = 10; % number of samples
sampling_spacing  = M/L;
data = generate_observations(x_true, N, noise_level, sampling_spacing);

%% EM

S = inv(SIGMA); % Note: S is the inverse of the covarince matrix
% maximal number of iteration for EM
niter = 1000; 
% tolerance for stopping criterion 
tolerance = 1e-5; 
% preparing variables for multiple initializations
x_est = zeros(M, num_EM_trial); 
MaxLL = zeros(num_EM_trial, 1);
% plot the Log Likelihood progress
flag_plot_LL = 0;
if flag_plot_LL 
figure; hold on;
end
EM_verbosity = 0;

for iter_em = 1:num_EM_trial
 if num_EM_trial>1
    fprintf('EM trial #%g out of %g\n',iter_em,num_EM_trial ); 
 end
 % initializing EM
x_init = mvnrnd(zeros(M,1), SIGMA);
x_init = x_init(:);
x_init = LP_proj(x_init, B);
% EM iterations
[x_est(:,iter_em), LL, LL_dis] = SR_EM(data, noise_level, sampling_spacing, x_init, S, B, niter,...
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

ln = 1.1; 
xn = x_true + noise_level*randn(M,1);
load('color');
figure; 
hold on;
plot(1:M, x_true,'linewidth',ln);
stem(1:2*B+1:M, xn(1:2*B+1:end),'square','linewidth',1.5);        
plot(1:M, LP_proj(x_true, B/2),'linewidth',ln, 'color',color(5,:));
plot(1:M, x_est_best,'linewidth',ln);
legend('true', 'samples', 'low-passed','estimated')
axis tight
set(gca,'Visible','off')
set(gcf,'color','w');
hold off;
fontsz = 11;
filename = 'Fig2.pdf';
pdf_print_code(gcf, filename, fontsz)

ln = 1.2;
figure;
err_fft = abs(fft(x_true) - fft(x_est_best))./abs(fft(x_true));
err_fft = err_fft(1:B+1);
hold on;
stem(0:B, err_fft);
plot([(L-1)/2, (L-1)/2],[0 max(err_fft)],'--','linewidth',ln)
ylabel('Relative error')
xlabel('frequency')
axis tight
filename = 'Fig2b.pdf';
pdf_print_code(gcf, filename, fontsz)

