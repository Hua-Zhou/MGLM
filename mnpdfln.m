function [logl] = mnpdfln(x,prob)
%MNPDFLN Multinomial log probability density function (pdf).
%   LOGL = MNPDFLN(X,PROB) returns the log-density for the multinomial
%   distribution with parameters PROB, evaluated at each row of X. X and
%   PROB are M-by-K matrices or 1-by-K vectors, where K is the number of
%   bins or categories.  Each element of PROB must be positive, and the
%   sample sizes for each observation (rows of X) are given by the row sums
%   SUM(X,2).  LOGL is a M-by-1 vector, and MNPDFLN computes each row of Y
%   using the corresponding rows of the inputs, or replicates them if
%   needed.
%
%   Examples
%    See documentation
%
%   See also MNFIT, MNRND, MNLOGITREG
%
%   Copyright 2012-2013 North Carolina State University 
%   Hua Zhou (hua_zhou@ncsu.edu) and Yiwen Zhang (yzhang31@ncsu.edu)

% check number of arguments
if nargin < 2
    error(message('mglm:mnpdfln:TooFewInputs'));
end

% If alpha is a column that can be interpreted as a single vector of
% Dir-Mult probabilities, transpose it to a row. Otherwise, treat as a
% matrix with one category. Transpose x if it is a non-empty column vector.
if size(prob,2)==1 && size(prob,1)>1
    prob = prob';
    if size(x,2)==1 && size(x,1)>1  % 
        x = x';
    end
end

% dimension match
[m,k] = size(x);
if k < 1
    error(message('mglm:mnpdfln:NoCategories'));
end
n = sum(x,2);   % batch sizes
[mm,kk] = size(prob);
if kk ~= k
    error('mglm:mnpdfln:InputSizeMismatch', ...
          'PROB and X must have the same number of columns.');
elseif mm == 1 % when m > 1
    prob = repmat(prob,m,1);
elseif m == 1
    m = mm;
    x = repmat(x,m,1);
    n = repmat(n,m,1);
elseif mm ~= m
    error('mglm:mnpdfln:InputSizeMismatch', ...
          ['PROB and X must have the same number of rows, '...
          'or either can be a row vector.']);
end

% now compute
logl = gammaln(n+1) - sum(gammaln(x+1),2) + sum(x.*log(prob),2);

end

