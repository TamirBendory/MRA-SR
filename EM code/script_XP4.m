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
seed = rng(3290);

L = 20; % signal length
K = 4; % down-sampling factor 
assert(mod(L,K)==0,'Please choose K to be a divsor of L');
Nyquist = (L/K-1)/2; % Nyquist sampling rate

% number of EM trials 
num_EM_trial = 5;

% Number of measurements
N = 1e9;

% Noise level
snr = 1/2;

%% Generating a signal with the desired power spectrum

%decaying rate of the signal's power spectrum
beta = 2; 
% strict bandwidth (only 2B-1 are non-zero)
B = 5;  

[x_true, sigma_f, SIGMA] = generate_signal(beta, L);

%projecting the signal into B low frequncies
x_true = LP_proj(x_true, B);

%% Generate the data

noise_level = norm(x_true)/sqrt(snr*L); % snr = norm(x)^2/(L*sigma^2)
save('parameters_XP2');
data = generate_observations(x_true, N, noise_level, K);
%% EM

S = inv(SIGMA); % Note: S is the inverse of the covarince matrix
% maximal number of iteration for EM
niter = 1000; 
% tolerance for stopping criterion 
tolerance = 1e-5; 

% preparing variables for multiple initializations
x_est_XP4 = zeros(L, num_EM_trial); 
MaxLL = zeros(num_EM_trial, 1);

EM_verbosity = 1;

for iter_em = 1:num_EM_trial
 
% initializing EM
x_init = mvnrnd(zeros(L,1), SIGMA);
x_init = x_init(:);
x_init = LP_proj(x_init, B);

[x_est_XP4(:,iter_em), LL, LL_dis] = SR_EM(data, noise_level, K, x_init, S, B, niter,...
    tolerance, EM_verbosity);

% maximal value of the log-likelihood function 
MaxLL(iter_em) = LL(end); 

save('x_est_XP4','x_est_XP4');

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

x_est_best = x_est_XP4(:,ind);
save('x_est_best_XP4','x_est_best');

%% Evaluate quality of recovery

x_est_best = align_to_reference(q1, x_true);
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
fontsz = 11;
filename = 'XP4.pdf';
pdf_print_code(gcf, filename, fontsz)