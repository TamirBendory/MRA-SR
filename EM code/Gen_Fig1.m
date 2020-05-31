clear;
close all;
clc;
seed = rng(101);

%% Problem setup
M = 120; % signal's length
L = 15;  %samples per observation
K = M/L; % down-sampling factor 
beta = 0; % i.i.d. normal signal

%% Generating signal
[x_true, sigma_f, SIGMA] = generate_signal(beta, M);
shift = 5;
x_true(round(M/2)-shift) = x_true(round(M/2)-shift) + 1;
x_true(round(M/2)+shift) = x_true(round(M/2)+shift) + 1;
x_true = LP_proj(x_true, L);

% Noise level
snr = 1;
noise_level = norm(x_true)/sqrt(snr*M); % snr = norm(x)^2/(M*sigma^2)

%% plotting

ln = 1.1; 
load('color');

circ_sh = -10;
xn = x_true + noise_level*randn(M,1);
figure; 
hold on;
plot(1:M, circshift(x_true, circ_sh),'linewidth',ln);
stem(1:K:M, circshift(xn(1:K:end),circ_sh) ,'--square','linewidth',ln)
axis tight
set(gca,'Visible','off')
set(gcf,'color','w');
hold off;
filename = 'Fig1a';
saveas(gcf, strcat(filename,'.fig'));
saveas(gcf, strcat(filename,'.jpg'));
fontsz = 11;
pdf_print_code(gcf, strcat(filename,'.pdf'), fontsz)

circ_sh = 20;
xn = x_true + noise_level*randn(M,1);
figure; 
hold on;
plot(1:M, circshift(x_true,circ_sh),'linewidth',ln);
stem(1:K:M, circshift(xn(1:K:end),circ_sh),'--square','linewidth',ln)
axis tight
set(gca,'Visible','off')
set(gcf,'color','w');
hold off;
filename = 'Fig1b';
saveas(gcf, strcat(filename,'.fig'));
saveas(gcf, strcat(filename,'.jpg'));
fontsz = 11;
pdf_print_code(gcf, strcat(filename,'.pdf'), fontsz)
