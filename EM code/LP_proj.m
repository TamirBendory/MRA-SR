function x_out = LP_proj(x_in, B)

% projects the signal x_in into its first B Fourier coefficients

L = size(x_in, 1);
x_out_fft = zeros(size(x_in));
xf = fft(x_in);
x_out_fft(1:B, :) = xf(1:B, :);
x_out_fft(L:-1:L-B+2,:) = xf(L:-1:L-B+2,:);
x_out = ifft(x_out_fft);

end