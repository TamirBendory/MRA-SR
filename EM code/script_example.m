clear;
close all;
clc;

%% Problem setup

% Start the parallel pool
parallel_nodes = 2;
if isempty(gcp('nocreate'))
    parpool(parallel_nodes, 'IdleTimeout', 240);
end

M = 100; %observation's length
L = 50;  %number of samples
K = M/L; % down-sampling factor

% number of EM trials
num_EM_trial = 50;
% Number of measurements
N = 1e5;
% Noise level
snr = 10;
flag_one_round = 0;

%% Generating a signal with the desired power spectrum

%decaying rate of the signal's power spectrum
beta = 1;
[x_true, sigma_f, SIGMA] = generate_signal(beta, M);

%% Generate the data

noise_level = 1/sqrt(snr*M); % snr = norm(x)^2/(M*sigma^2)
%save('parameters');
data = generate_observations(x_true, N, noise_level, K);

%% EM

% maximal number of iteration for EM
options.niter = 100;
% tolerance for stopping criterion
options.tolerance = 1e-5;
% verbosity
options.verbosity = 1;
% preparing variables for multiple initializations
x_est = zeros(M, num_EM_trial);
x_est_low =  zeros(L, num_EM_trial);
MaxPosterior = zeros(num_EM_trial, 1);

for iter_em = 1:num_EM_trial
    
    if num_EM_trial>1
        fprintf('EM trial #%g out of %g\n',iter_em, num_EM_trial );
    end
    % initializing EM
    x_init = mvnrnd(zeros(M,1), SIGMA);
    x_init = x_init(:);
        %first round
    if flag_one_round == 0
    x_init = decimate(x_init, K);
    [~, ~, SIGMA_low] = generate_signal(beta, L);
    options.S = inv(SIGMA_low);
    [x_est_low(:,iter_em), ~, ~] = SR_EM(data, noise_level, 1, x_init, [], options);
     x_init = interp(x_est_low(:,iter_em), K);
    end
    %second round
    options.S = inv(SIGMA);
    [x_est(:,iter_em), post_value, post_dis] = SR_EM(data, noise_level, K,x_init , [], options);
    
end

% maximal value of the log-likelihood function
MaxPosterior(iter_em) = post_value(end);

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
subplot(2,1,2);
hold on;
plot(err_fft);
%plot([Nyquist Nyquist],[0 max(err_fft)],'--')
title('Relative error as a function of frequency')
axis tight
fontsz = 11;
filename = 'script_example.pdf';
%pdf_print_code(gcf, filename, fontsz)