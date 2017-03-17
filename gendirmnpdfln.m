function [logl] = gendirmnpdfln(x,alpha,beta)
%GENDIRMNPDFLN Generalized Dichlet-Multinomial log prob. density function
%   Y = GENDIRMNPDFLN(X,ALPHA,BETA) returns the log-density for the
%   generalized Dirichlet-Multinomial distribution with parameters ALPHA
%   and BETA, evaluated at each row of X. X is M-by-D matrices or 1-by-D
%   vectors, where D is the number of bins or categories.  Each element of
%   ALPHA and BETA must be positive, and the sample sizes for each
%   observation (rows of X) are given by the row sums SUM(X,2).  LOGL is a
%   M-by-1 vector, and gendirmnpdfln computes each row of LOGL using the
%   corresponding rows of the inputs, or replicates them if needed.
%
%   INPUT:
%       X: n-by-d count matrix
%       alpha: 1-by-(d-1) or n-by-(d-1) parameter values
%       beta: 1-by-(d-1) or n-by-(d-1) parameter values
%
%   OUTPUT:
%       logl: n-by-1 log probability densities for each row of X
%
% Example:
%   See documentation
%
% See also GENDIRMNFIT, GENDIRMNRND, GENDIRMNREG
%
% Copyright 2015-2017 University of California at Los Angeles
% Hua Zhou (hua_zhou@ncsu.edu)

% check number of arguments
if nargin < 3
    error(message('mglm:gendirmnpdfln:TooFewInputs'));
end

% If alpha is a column that can be interpreted as a single vector of
% Dir-Mult probabilities, transpose it to a row. Otherwise, treat as a
% matrix with one category. Transpose x if it is a non-empty column vector.
if size(alpha,2)==1 && size(alpha,1)>1
    alpha = alpha';
    if size(x,2)==1 && size(x,1)>1  % 
        x = x';
    end
end
if size(beta,2)==1 && size(beta,1)>1
    beta = beta';
end

% dimension match
[m,k] = size(x);
if k < 1
    error(message('mglm:gendirmnpdfln:NoCategories'));
end
n = sum(x,2);   % batch sizes
[mm,kk] = size(alpha);
[mm2,kk2] = size(beta);
if kk~=k-1 || kk2~=k-1
    error('mglm:gendirmnpdfln:InputSizeMismatch', ...
          'ALPHA and BETA must have d-1 columns.');
elseif mm == 1 && mm2 == 1 % when m > 1
    alpha = repmat(alpha,m,1);
    beta = repmat(beta,m,1);
elseif m == 1
    m = mm;
    x = repmat(x,m,1);
    n = repmat(n,m,1);
elseif mm ~= m
    error('mglm:gendirmnpdfln:InputSizeMismatch', ...
          'ALPHA and X must have the same number of rows, or either can be a row vector.');
end

% now compute
z = bsxfun(@minus, n, [zeros(m,1) cumsum(x(:,1:end-1),2)]);
logl = gammaln(n+1) - sum(gammaln(x+1),2);
logl = logl + sum(gammaln(x(:,1:k-1)+alpha),2) - sum(gammaln(alpha),2) ...
    + sum(gammaln(z(:,2:k)+beta),2) - sum(gammaln(beta),2) ...
    - sum(gammaln(alpha+beta+z(:,1:k-1)),2) + sum(gammaln(alpha+beta),2);

end
