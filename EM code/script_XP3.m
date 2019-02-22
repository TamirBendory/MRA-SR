clear;
close all;
clc;

%% Problem setup

% Start the parallel pool
parallel_nodes = 20;
if isempty(gcp('nocreate'))
    parpool(parallel_nodes, 'IdleTimeout', 240);
end

% seeds the random number generator
 seed = rng(653);

L = 60; % signal length
K = 6; % down-sampling factor 
assert(mod(L,K)==0,'Please choose K to be a divsor of L');

% number of EM trials 
num_EM_trial = 100;

% Number of measurements
N = 1e3;

% Noise level
snr = 10;

%decaying rate of the signal's power spectrum
beta = 1;

%% Loop over different values of B
num_iter = 100;
B_vec = 6:2:40;

err_x_XP3 = zeros(num_iter, length(B_vec));

for iter = 1:num_iter
    for b = 1:length(B_vec)
        
B = B_vec(b);  

[x_true, sigma_f, SIGMA] = generate_signal(beta, L);
x_true = LP_proj(x_true, B);

%% Generate the data

noise_level = norm(x_true)/sqrt(snr*L); % snr = norm(x)^2/(L*sigma^2)
data = generate_observations(x_true, N, noise_level, K);
%% EM

S = inv(SIGMA); % Note: S is the inverse of the covarince matrix
% maximal number of iteration for EM
niter = 1000; 
% tolerance for stopping criterion 
tolerance = 1e-5; 

% preparing variables for multiple initializations
x_est = zeros(L, num_EM_trial); 
MaxLL = zeros(num_EM_trial, 1);

EM_verbosity = 0;

for iter_em = 1:num_EM_trial

 if num_EM_trial>1
    fprintf('EM trial #%g out of %g\n',iter_em,num_EM_trial ); 
 end
 
% initializing EM
x_init = mvnrnd(zeros(L,1), SIGMA);
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

x_est_best = align_to_reference(x_est_best, x_true);
err_x_XP3(iter,b) = norm(x_est_best - x_true) / norm(x_true);
fprintf('recovery error = %.4g\n', err_x_XP3(iter,b));

    end
    save('err_x_XP3','err_x_XP3');
end

%% plotting

err_x_XP3 = mean(err_x_XP3,1);

figure; 
hold on;
%set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
plot(B_vec, err_x_XP3,'linewidth',0.8);
xlim([B_vec(1),B_vec(end)]);
ylim([0.01,1]);
grid on;
%axis tight;
%line([vert_line vert_line],get(gca,'YLim'),'Color','red','LineStyle', '--')
xlabel('B');
ylabel('relative error');

fontsz = 11;
filename = 'XP3.pdf';
pdf_print_code(gcf, filename, fontsz)