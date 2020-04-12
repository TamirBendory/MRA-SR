clear;
close all;
clc;
%dbstop if error

%% Problem setup

% Start the parallel pool
% parallel_nodes = 2;
% if isempty(gcp('nocreate'))
%     parpool(parallel_nodes, 'IdleTimeout', 240);
% end

% seeds the random number generator
seed = rng(9843);

B = 10;  % strict bandwidth (only 2B+1 are non-zero)
L = 10*(2*B+1); % signal length

%decaying rate of the signal's power spectrum
beta = 0; 
[x_LP, sigma_f, SIGMA] = generate_signal(beta, L);
x_LP = LP_proj(x_LP, B/2); %bandwidth of B=5 ;

x_true = x_LP;
shift = 10;
x_true(round(L/2)-shift) = x_true(round(L/2)-shift) + 0.5;
x_true(round(L/2)+shift) = x_true(round(L/2)+shift) + 0.5;
x_true = LP_proj(x_true, B);

% Noise level
snr = 1/2;
noise_level = norm(x_true)/sqrt(snr*L); % snr = norm(x)^2/(L*sigma^2)

% number of EM trials 
num_EM_trial = 5;

% Number of measurements
N = 1e4;

%% Generate the data

factor = 2; % the down-sampling ratio
sampling_spacing = L/(2*B+1)*factor+1; % sampling spacing
data = generate_observations(x_true, N, noise_level, sampling_spacing);

%% EM

S = inv(SIGMA); % Note: S is the inverse of the covarince matrix
% maximal number of iteration for EM
niter = 1000; 
% tolerance for stopping criterion 
tolerance = 1e-5; 

% preparing variables for multiple initializations
x_est = zeros(L, num_EM_trial); 
MaxLL = zeros(num_EM_trial, 1);

% plot the Log Likelihood progress
flag_plot_LL = 0;
if flag_plot_LL 
figure; hold on;
end

EM_verbosity = 1;
for iter_em = 1:num_EM_trial

 if num_EM_trial>1
    fprintf('EM trial #%g out of %g\n',iter_em,num_EM_trial ); 
 end
 % initializing EM
x_init = mvnrnd(zeros(L,1), SIGMA);
x_init = x_init(:);
x_init = LP_proj(x_init, B);

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
%x_est_best = x_est_best/mean(x_est_best)*mean(data(:));
save('x_est_best','x_est_best');

%% Evaluate quality of recovery

x_est_best = align_to_reference(x_est_best, x_true);
err_x = norm(x_est_best - x_true) / norm(x_true);
fprintf('recovery error = %.4g\n', err_x);

ln = 1.1; 
xn = x_true + noise_level*randn(L,1);
load('color');
figure; 

%subplot(2,1,1);
hold on;
plot(1:L, x_true,'linewidth',ln);
stem(1:2*B+1:L, xn(1:2*B+1:end),'square','linewidth',1.5);        
plot(1:L, LP_proj(x_true, B/2),'linewidth',ln, 'color',color(5,:));
plot(1:L, x_est_best,'linewidth',ln);
legend('true', 'samples', 'low-passed','estimated')
%title(strcat('err = ', num2str( err_x)));
axis tight
set(gca,'Visible','off')
set(gcf,'color','w');
hold off;

% err_fft = abs(fft(x_true) - fft(x_est_best))./abs(fft(x_true));
% err_fft = err_fft(1:B);
% subplot(2,1,2); 
% hold on;
% plot(1:B, err_fft);
% plot([Nyquist Nyquist],[0 max(err_fft)],'--')
% title('Relative error as a function of frequency')
% axis tight
fontsz = 11;
filename = 'XP5.pdf';
pdf_print_code(gcf, filename, fontsz)