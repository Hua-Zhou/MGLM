function [logl] = negmnpdfln(x,prob,b)
%NEGMNPDFLN Negative multinomial log probability density function (pdf)
%   Y = NEGMNPDFLN(X,PROB,B) returns the log-density for the negative
%   multinomial distribution with probability parameter PROB and
%   over-dispersion parameter B, evaluated at each row of X. X and PROB are
%   M-by-D matrices or 1-by-D vectors, where D is the number of bins or
%   categories. B is M-by-1 vector or a scalar Each element of B must be
%   positive, and the sample sizes for each observation (rows of X) are
%   given by the row sums SUM(X,2).  LOGL is a M-by-1 vector, and
%   negmnpdfln computes each element of LOGL using the corresponding rows
%   of the inputs, or replicates them if needed.
%
%   INPUT:
%       X: n-by-d count matrix
%       prob: 1-by-(d+1) or n-by-(d+1) parameter values
%       b: scalar or n-by-1 parameter values
%
%   OUTPUT:
%       logl: n-by-1 log probability densities for each row of X

%   Example
%    See documentation
%
%   See also NEGMNFIT, NEGMNREG, NEGMNRND, NEGMNREG2
%
% Copyright 2015-2017 University of California at Los Angeles
% Hua Zhou (hua_zhou@ncsu.edu)

% check number of arguments
if nargin < 2
    error(message('mglm:dirmultpdfln:TooFewInputs'));
end

% If PROB is a column that can be interpreted as a single vector of
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
    error(message('mglm:negmultpdfln:NoCategories'));
end
[mm,kk] = size(prob);
if (size(b,1)~=mm)
    error('mglm:negmultpdfln:InputSizeMismatch', ...
        'PROB and B should have same number of rows.');
end    
if kk ~= k+1
    error('mglm:dirmultpdfln:InputSizeMismatch', ...
          'Number of columns PROB should be that of X plus one.');
elseif mm == 1 % when m > 1
    prob = repmat(prob,m,1);
    b = repmat(b,m,1);
elseif m == 1
    m = mm;
    x = repmat(x,m,1);
elseif mm ~= m
    error('mglm:dirmultpdfln:InputSizeMismatch', ...
          'PROB and X must have the same number of rows, or either can be a row vector.');
end

% now compute
logl = gammaln(b+sum(x,2)) - gammaln(b) - sum(gammaln(x+1),2) ...
    + sum(x.*log(prob(:,1:(end-1))),2) + b.*log(prob(:,end));

end
