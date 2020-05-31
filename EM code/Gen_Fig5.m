clear;
close all;
clc;
seed = rng(4383);

% parallel_nodes = 4;
% if isempty(gcp('nocreate'))
%     parpool(parallel_nodes, 'IdleTimeout', 240);
% end

%% Problem setup

M = 60; % signal's length
%M = 120;
%M = 240;
% Number of observations
N = 10^3; % % number of observations (high SNR)
beta = 1;
snr = 5;
noise_level = norm(x_true)/sqrt(snr*M); 

%% EM parameters

num_EM_trial = 10; % % number of EM trials (high SNR)
options.niter = 100; % maximal number of iteration for EM
options.tolerance = 1e-5; % tolerance for stopping criterion
options.verbosity = 0; % verbosity
options.S = inv(SIGMA);

%% a loop over multiple L values

num_iter = 50;
L_vec = [5,10, 15, 20, 30]; % M = 60
%L_vec = [10, 15, 25,30,40,60]; % M = 120
%L_vec = [5,10, 15, 20, 30, 40, 60, 80, 120]; % M = 240
length_L = length(L_vec);
error = zeros(num_iter, length_L);
[x_true, sigma_f, SIGMA] = generate_signal(beta, M); % the signal is generated only once

for iter = 1:num_iter
    for l = 1:length_L
        L = L_vec(l);
        K = round(M/L); % in this experiment, we allow M/L to be non-integer
        fprintf('iter = %.0g,  L = %g\n\n',iter, L);
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
            [x_est(:,iter_em), post_value, post_dis] = SR_EM(data, noise_level, K,x_init , [], options);
            MaxPosterior(iter_em) = post_value(end);
        end
        
        % choosing the "best" signal among all trials
        [~, ind] = max(MaxPosterior);
        x_est_best = x_est(:,ind);
        
        %% Evaluate quality of recovery
        x_est_best = align_to_reference(x_est_best, x_true);
        error(iter,l) = norm(x_est_best - x_true) / norm(x_true);
        fprintf('recovery error = %.4g\n', error(iter,l));        
    end
    save('err_Fig5','error');
end

%% plotting

err_x = median(error, 1);
var_x = var(error, 1);
ln = 1.1;
figure; 
hold on;
stem(L_vec, err_x,'linewidth', ln);
xline(M^(2/3),'--','color','red','linewidth', ln);
set(gca, 'YScale', 'log')
%set(gca, 'XScale', 'log')
%xlabel('SNR');
xlabel('L');
ylabel('relative error');
grid on;
axis tight;
ylim([0,1])
%xlim([1,max(L_vec)+1])

filename = 'Fig5_M60.pdf';
%filename = 'Fig5_M120.pdf';
%filename = 'Fig5_M240.pdf';
saveas(gcf, strcat(filename,'.fig'));
saveas(gcf, strcat(filename,'.jpg'));
fontsz = 11;
pdf_print_code(gcf, strcat(filename,'.pdf'), fontsz)
