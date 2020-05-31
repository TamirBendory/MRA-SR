clear;
close all;
clc;
seed = rng(637);

% parallel_nodes = 4;
% if isempty(gcp('nocreate'))
%     parpool(parallel_nodes, 'IdleTimeout', 240);
% end
%
%% Problem setup

M = 64; % signal's length
L = 32;  %samples per observation
K = M/L; % down-sampling factor
beta = 1; % spectrum dacys as 1/f

% number of observations
N = 10^3; %high SNR
%N = 10^5; %low SNR

% number of EM trials
num_EM_trial = 1000; %high SNR
%num_EM_trial = 20; %low SNR


%% a loop over multiple snr values
num_iter = 100;
length_snr_vec = 30;
snr_vec = logspace(0, 0.6, length_snr_vec); %high SNR
%snr_vec = logspace(-0.5, 0.4, length_snr_vec); %low SNR
error_exp_snr = zeros(num_iter, length_snr_vec);

% EM parameters
options.niter = 100; % maximal number of iteration for EM
options.tolerance = 1e-5; % tolerance for stopping criterion
options.verbosity = 0; % verbosity

for iter = 1:num_iter
    for s = 1:length_snr_vec
        
        snr = snr_vec(s);
        fprintf('iter = %.0g,  snr = %.2g\n\n\n\n',iter, snr );
        [x_true, sigma_f, SIGMA] = generate_signal(beta, M);
        options.S = inv(SIGMA);
        noise_level = norm(x_true)/sqrt(snr*M); % snr = norm(x)^2/(L*sigma^2)
        data = generate_observations(x_true, N, noise_level, K);
        
        %% EM
        x_est = zeros(M, num_EM_trial);
        MaxPosterior = zeros(num_EM_trial, 1);
        
        for iter_em = 1:num_EM_trial
            if num_EM_trial>1
                fprintf('EM trial #%g out of %g\n',iter_em,num_EM_trial );
            end
            x_init = mvnrnd(zeros(M,1), SIGMA);
            x_init = x_init(:);
            options.S = inv(SIGMA);
            [x_est(:,iter_em), post_value, post_dis] = SR_EM(data, noise_level, K,x_init , [], options);
            MaxPosterior(iter_em) = post_value(end);
        end
        % choosing the "best" signal among all trials
        [~, ind] = max(MaxPosterior);
        x_est_best = x_est(:,ind);
        
        %% Evaluate quality of recovery
        
        x_est_best = align_to_reference(x_est_best, x_true);
        error_exp_snr(iter,s) = norm(x_est_best - x_true) / norm(x_true);
        fprintf('recovery error = %.4g\n', error_exp_snr(iter,s));
        
    end
    save('error_exp_snr','error_exp_snr');
end

%% plotting

err_x = median(error_exp_snr, 1);
var_x = var(error_exp_snr, 1);

figure; hold on;
plot(snr_vec, err_x);
%errorbar(snr_vec, err_x, var_x);
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
xlabel('SNR');
ylabel('relative error');
grid on;
axis tight;

%filename = 'Fig3_low_snr';
filename = 'Fig3_high_snr';
saveas(gcf, strcat(filename,'.fig'));
saveas(gcf, strcat(filename,'.jpg'));
fontsz = 11;
pdf_print_code(gcf, strcat(filename,'.pdf'), fontsz)
