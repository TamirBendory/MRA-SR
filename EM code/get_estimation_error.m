function [R,x_hat] = get_estimation_error(s,x,K)
    
    N = length(s);
    
    
    % convert signal to matrix    
    S = reshape(s,K,N/K)';
    X = reshape(x,K,N/K)';
    
    % measure pairwise distance
    D = zeros(K);
    n_S = vecnorm(S).^2;
    n_X = vecnorm(X).^2;    
    for i = 1:K
        for j = 1:K
            D(j,i) = n_S(i)+n_X(j)-2*max(cconv(S(:,i),flipud(X(:,j)),N/K));
        end
    end
    
    % check all permutations
    [Idx,R] = hungarian(D);
    X = X(:,Idx);
    
    % align to ref
    for i = 1:K
       X(:,i) = align_to_reference(X(:,i),S(:,i)); 
    end
    x_hat = reshape(X',1,N);    
    
end