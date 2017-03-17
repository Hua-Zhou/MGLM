function [logl] = dirmnpdfln(x,alpha)
%DIRMNPDFLN Dichlet-Multinomial log probability density function (pdf)
%   Y = DIRMNPDFLN(X,ALPHA) returns the log-density for the
%   Dirichlet-Multinomial distribution with parameters ALPHA, evaluated at
%   each row of X. X and ALPHA are M-by-K matrices or 1-by-K vectors, where
%   K is the number of bins or categories.  Each element of ALPHA must be
%   positive, and the sample sizes for each observation (rows of X) are
%   given by the row sums SUM(X,2).  LOGL is a M-by-1 vector, and
%   DIRMNPDFLN computes each element of Y using the corresponding rows of
%   the inputs, or replicates them if needed.
%
%   INPUT:
%       X: n-by-d count matrix
%       alpha: 1-by-d or n-by-d parameter values
%
%   OUTPUT:
%       logl: n-by-1 log probability densities for each row of X
%
% Examples
%   See documentation
%
% See also DIRMNFIT, DIRMNRND, DIRMNREG
%
% Copyright 2015-2017 University of California at Los Angeles
% Hua Zhou (hua_zhou@ncsu.edu)

% check number of arguments
if nargin < 2
    error(message('mglm:dirmultpdfln:TooFewInputs'));
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

% dimension match
[m,k] = size(x);
if k < 1
    error(message('mglm:dirmultpdfln:NoCategories'));
end
n = sum(x,2);   % batch sizes
[mm,kk] = size(alpha);
if kk ~= k
    error('mglm:dirmultpdfln:InputSizeMismatch', ...
          'ALPHA and X must have the same number of columns.');
elseif mm == 1 % when m > 1
    alpha = repmat(alpha,m,1);
elseif m == 1
    m = mm;
    x = repmat(x,m,1);
    n = repmat(n,m,1);
elseif mm ~= m
    error('mglm:dirmultpdfln:InputSizeMismatch', ...
          ['ALPHA and X must have the same number of rows, '...
          'or either can be a row vector.']);
end

% now compute
alpha_rowsums = sum(alpha,2);
logl = gammaln(n+1) - sum(gammaln(x+1),2);
logl = logl + sum(gammaln(x+alpha),2) - sum(gammaln(alpha),2) ...
    + gammaln(alpha_rowsums) - gammaln(alpha_rowsums+n);

end

