clear;
close all;
clc;
seed = rng(101);

% Start the parallel pool
% parallel_nodes = 4;
% if isempty(gcp('nocreate'))
%     parpool(parallel_nodes, 'IdleTimeout', 240);
% end

%% Problem setup

M = 120; % signal's length
L = 15;  %samples per observation
K = M/L; % down-sampling factor 
beta = 0; % i.i.d. normal signal
snr = 1; % Noise level
num_EM_trial = 5; % number of EM trials 
N = 1e4; % Number of observations

%% Generating signal

[x_true, sigma_f, SIGMA] = generate_signal(beta, M);
shift = 5;
x_true(round(M/2)-shift) = x_true(round(M/2)-shift) + 1;
x_true(round(M/2)+shift) = x_true(round(M/2)+shift) + 1;
x_true = LP_proj(x_true, L);
% A low-passed version of the signal; used only for visualization
x_true_LP = LP_proj(x_true, floor(L/2)); 

%% Generating data

noise_level = norm(x_true)/sqrt(snr*M); % snr = norm(x)^2/(M*sigma^2)
data = generate_observations(x_true, N, noise_level, K);

%% EM

options.S = inv(SIGMA);
options.niter = 100; % maximal number of iteration for EM
options.tolerance = 1e-5; % tolerance for stopping criterion 
options.verbosity = 1; % verbosity
% preparing variables for multiple initializations
x_est = zeros(M, num_EM_trial); 
MaxPosterior = zeros(num_EM_trial, 1);

for iter_em = 1:num_EM_trial
 if num_EM_trial>1
    fprintf('EM trial #%g out of %g\n',iter_em, num_EM_trial ); 
 end
x_init = mvnrnd(zeros(M,1), SIGMA);
x_init = x_init(:);
[x_est(:,iter_em), post_value, post_dis] = SR_EM(data, noise_level, K, x_init, L, options);
MaxPosterior(iter_em) = post_value(end); 
end

% Choosing the "best" signal among all trials
[~, ind] = max(MaxPosterior);
x_est_best = LP_proj(x_est(:,ind), L);
%save('x_est_best','x_est_best');
x_est_best = align_to_reference(x_est_best, x_true);
err_x = norm(x_est_best - x_true) / norm(x_true);
fprintf('recovery error = %.4g\n', err_x);

%% Plotting

ln = 1.1; 
load('color');
figure; 
hold on;
plot(1:M, x_true,'linewidth',ln,'linestyle','--');
plot(1:M, x_true_LP, 'linewidth',ln);
plot(1:M, x_est_best,'linewidth',ln,'color',color(5,:));
legend('true', 'low-passed','estimated')
axis tight
set(gca,'Visible','off')
set(gcf,'color','w');
hold off;
filename = 'Fig2';
saveas(gcf, strcat(filename,'.fig'));
saveas(gcf, strcat(filename,'.jpg'));
fontsz = 11;
pdf_print_code(gcf, strcat(filename,'.pdf'), fontsz)

ln = 1.1;
figure;
err_fft = abs(fft(x_true) - fft(x_est_best))./abs(fft(x_true));
err_fft = err_fft(1:L+1);
hold on;
stem(0:L, err_fft);
plot([L/2, L/2],[0 max(err_fft)],'--','linewidth',ln)
ylabel('Relative error')
xlabel('frequency')
axis tight
filename = 'Fig2b';
saveas(gcf, strcat(filename,'.fig'));
saveas(gcf, strcat(filename,'.jpg'));
fontsz = 11;
pdf_print_code(gcf, strcat(filename,'.pdf'), fontsz)
