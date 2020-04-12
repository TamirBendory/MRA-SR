clear;
close all;
clc;
dbstop if error
% seeds the random number generator
seed = rng(7483);

% Start the parallel pool
parallel_nodes = 2;
if isempty(gcp('nocreate'))
    parpool(parallel_nodes, 'IdleTimeout', 240);
end

%% Problem setup

% signal's parameters
B = 5;  
L = 5;
M = L*(2*B+1); % signal length
%M = 40; % signal length
K = M/L;% down-sampling factor 
beta = 1; 
% number of EM trials 
num_EM_trial = 500;
% Number of measurements
N = 1e4;
% maximal number of iteration for EM
niter = 100; 
% tolerance for stopping criterion 
tolerance = 1e-5; 

%% a loop over multiple snr values

num_iter = 5; %50;
length_snr_vec = 10; %20;
snr_vec = logspace(-0.2,1.5,length_snr_vec);
err_x_XP1 = zeros(num_iter, length_snr_vec);

for iter = 1:num_iter
    for s = 1:length_snr_vec
    
[x_true, sigma_f, SIGMA] = generate_signal(beta, M);
x_true = LP_proj(x_true, B);
snr = snr_vec(s);
noise_level = norm(x_true)/sqrt(snr*M); % snr = norm(x)^2/(L*sigma^2)
data = generate_observations(x_true, N, noise_level, K);

%% EM
% preparing variables for multiple initializations
x_est = zeros(M, num_EM_trial); 
MaxLL = zeros(num_EM_trial, 1);
S = inv(SIGMA); 
EM_verbosity = 0;

for iter_em = 1:num_EM_trial
 if num_EM_trial>1
    fprintf('EM trial #%g out of %g\n',iter_em,num_EM_trial ); 
 end
 
% initializing EM
x_init = mvnrnd(zeros(M,1), SIGMA);
x_init = x_init(:);
x_init = LP_proj(x_init, B);
[x_est(:,iter_em), LL, LL_dis] = SR_EM(data, noise_level, K, x_init, S, B, niter,...
    tolerance, EM_verbosity);
% maximal value of the log-likelihood function 
MaxLL(iter_em) = LL(end); 
end

% choosing the "best" signal among all trials
[~, ind] = max(MaxLL);
x_est_best = x_est(:,ind);

%% Evaluate quality of recovery

x_est_best = align_to_reference(x_est_best, x_true);
err_x_XP1(iter,s) = norm(x_est_best - x_true) / norm(x_true);
fprintf('recovery error = %.4g\n', err_x_XP1(iter,s));

    end
    save('err_x_XP1','err_x_XP1');
end

%% plotting 

%load('err_x_XP1');
err_x = mean(err_x_XP1,1);

figure; hold on;
plot(snr_vec(2:end), err_x(2:end));
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
xlabel('SNR');
ylabel('relative error');
grid on;
axis tight;

if 0
slope_high_snr = (log(err_x(end)) - log(err_x(12)))/(log(snr_vec(end)) - log(snr_vec(12)));
slope_low_snr = (log(err_x(7)) - log(err_x(2)))/(log(snr_vec(7)) - log(snr_vec(2)));

 fontsz = 11;
 filename = 'Fig3.pdf';
 pdf_print_code(gcf, filename, fontsz)
end