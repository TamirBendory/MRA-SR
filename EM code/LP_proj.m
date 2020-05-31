function x_out = LP_proj(x_in, B)

% Projects the signal x_in onto its first B Fourier coefficients, while
% keeping the signal real valued

M = size(x_in, 1);
x_out_fft = zeros(size(x_in));
xf = fft(x_in);
x_out_fft(1:B+1, :) = xf(1:B+1, :);
x_out_fft(M:-1:M-B+1,:) = xf(M:-1:M-B+1,:);
x_out = ifft(x_out_fft);

end