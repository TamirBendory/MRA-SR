clear all; 
close all;
clc;

% Parameters
K = 2;       % Subsampling rate
sigma_vec = exp(-3:0.5:0);  % noise level
N = 12;      % signal length
if isinteger(N/K)
fprintf('K does not divide N!\n')
end
m = 10^3;    % number of repetitions
tol = 10^-4; %
num_runs = 1;
max_itr = 10^3; %maximum EM iterations

% generate a random Gaussian signal with specified covariance 
%s = randn(N,1);
s = mvnrnd(zeros(N,1),eye(N));

R = zeros(num_runs,length(sigma_vec));
num_itr = zeros(num_runs,length(sigma_vec));
for j = 1:num_runs
    fprintf('%d:',j);
    
    %Go over values of sigma
    for sigma_idx = 1:length(sigma_vec)
        fprintf('%d ',sigma_idx);
        sigma = sigma_vec(sigma_idx);
                
        % Create noisy subsampled signals with low resolution
        X = zeros(N/K,m);
        idx_A = zeros(1,m);
        idx_B = zeros(1,m);
        for i = 1:m            
            %x_hat = s(randi(K):K:end);            
            %X(:,i) = circshift(x_hat,randi(N/K));
            X(:,i) = circshift(s(randi(K):K:end),randi(N/K));
        end        
        X = X + sigma*randn(N/K,m);
        
        x_init = randn(N,1);
        [x_hat,num_itr(j,sigma_idx)] = ...
            recover_signal_em_time_domain(X,K,sigma,x_init,tol,max_itr);
                
        % align signal matrix to estimated matrix and compute error
        [R(j,sigma_idx),x_aligned] = get_estimation_error(s,x_hat,K);
        
        if 0 
            fig = figure;
            plot(1:length(x_aligned),x_aligned,'s'); grid on;
            hold on;    
            plot(0.1+(1:length(s)),s,'sr');
            fprintf('relative error: %f\n',norm(x_aligned-s)/norm(s)); 
        end
        
    end
    fprintf('\n');
end

fig = figure;
loglog(sigma_vec,median(R,1),'-s','linewidth',2);grid on;
fig.CurrentAxes.FontSize = 12;
xlabel('Noise level $\sigma$','fontsize',12,'interpreter','latex');
ylabel( ' $l_2$ Error' ,'fontsize',12,'interpreter','latex');
print(fig,'em_reconstruction_error_vs_sigma.eps','-depsc');