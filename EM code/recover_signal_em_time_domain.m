function [x_hat, iter] = recover_signal_em_time_domain(X,K,sigma,x_init,tol,max_itr)
    
    % Expectation maximization algorithm for super resolution multireference
% allignment
% X: data - each column is an observation with low resolution
% sigma: noise standard deviation affecting measurements
% x: initial guess for the signal (optional)
% tol: EM stops iterating if two subsequent iterations are closer than tol
%      in 2-norm, up to circular shift (default: 1e-5).
% batch_niter: number of batch iterations to perform before doing full data
%              iterations if there are more than 3000 observations
%              (default: 3000.)
%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

    % X contains M observations, each of length N/S where S is the
    
    
    % subsampling parameter
    n = size(X,1);
    N = n*K;
        
    if ~exist('x_init', 'var') || isempty(x_init)
        if isreal(X)
            x_hat = randn(n, 1);
        else
            x_hat = randn(n, 1) + 1i*randn(n, 1);
        end
    else
        x_hat = x_init;
    end
    
    for iter = 1 : max_itr
        
        x_new = EM_iteration(K,x_hat, X, sigma);
        
        % Change this
        if sum(isnan(x_new))
           x_new = randn(N,1); 
        end
        
        % get residual by summing all subsampled parts of the signal
        res = relative_error(x_new,x_hat);
        
        
        if  res < tol
            break;
        end
        
        x_hat = x_new;
            
    end
    
end


% A single EM iteration
function x_new = EM_iteration(K,x_hat, X, sigma)
    
    n = size(X,1);
    N = K*n;    
    
    % E - step: calculate probabilites
        
    %pad X_til with (k-1) zeros
    X_til = reshape([X(:)'; zeros(K-1,size(X,2)*n)],N,size(X,2));
    
    %calculate vector norms
    alpha = repmat(vecnorm(X).^2,N,1);
    beta = zeros(N,size(X,2));
    for i = 1:K
       beta((K+1-i):K:end,:) = norm(x_hat(i:K:end))^2; 
    end
    
    % calculate the circular convolution 
    C = zeros(N,size(X,2));
    for i = 1:size(X,2)        
       C(:,i) = cconv(x_hat,flipud(X_til(:,i)),N);        
       %C(:,i) = cconv(x_hat,X_til(:,i),n);  
    end
    
    %obtain probabilities
    A = -1*(alpha+beta -2*C);
    
    % Ariel check this!
    A = A - max(A,[],1);
    %sigma_hat = sqrt(median(min(abs(A),[],1))/m); 
    
    W = exp( A/(2*sigma^2));
    W = W./sum(W,1);
    
    % M - step: re-calculate estimated signal
    
    % Calculate probabilites for each of the K sampling 
    P = zeros(K,size(X,2));
    W_hat = W;
    for i = 1:K
        P(i,:) = sum(W(i:K:end,:),1);
        
        % Update probabilites 
        %n_z_idx = find(P(i,:)>0);
        W_hat(i:K:end,P(i,:)>0) = W_hat(i:K:end,P(i,:)>0)./P(i,P(i,:)>0);
    end
    
    % get weighted average 
    X_new = zeros(N,size(X,2));
    for i = 1:size(X,2)
        %X_new(:,i) = cconv(W(:,i),X_til(:,i),n);
        X_new(:,i) = cconv(W_hat(:,i),X_til(:,i),N);
    end
    
    x_new = zeros(N,1);
    for i = 1:N
        x_new(i) = P(K-mod(i,K),:)*X_new(i,:)'./sum(P(K-mod(i,K),:));
    end
end