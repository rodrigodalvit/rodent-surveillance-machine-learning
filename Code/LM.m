function Xl = LM(I, m)
% Computes 2D Legendre moments up to order m of a grayscale image I
% Input:
%   I - grayscale image (assumed square for simplicity)
%   m - maximum order of Legendre moments
% Output:
%   Xl - matrix of Legendre moments of size m x m
%%
% Ensure square image
[Nr, Nc] = size(I);
if Nr ~= Nc
    error('Input image must be square for this implementation.');
end
N = Nr;

% Normalize coordinates from -1 to 1
xi = linspace(-1, 1, N);

% Compute 1D Legendre polynomials using recurrence
P = zeros(m+1, N);
P(1,:) = 1;
P(2,:) = xi;
for p = 2:m
    for k = 1:N
        P(p+1,k) = ((2*p - 1)*xi(k)*P(p,k) - (p - 1)*P(p-1,k)) / p;
    end
end

% Initialize output matrix
Xl = zeros(m, m);

% Compute 2D Legendre moments
for p = 1:m
    for q = 1:m
        Ppq = 0;
        for i = 1:N
            for j = 1:N
                Ppq = Ppq + P(p,i) * P(q,j) * double(I(i,j));
            end
        end
        Xl(p,q) = (2*p + 1) * (2*q + 1) * Ppq / (N^2);
    end
end
end
