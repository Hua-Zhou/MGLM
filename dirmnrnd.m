function [r] = dirmnrnd(n,alpha,m)
%DIRMNRND Random vectors from the Dirichlet-Multinomial distribution
%
%   R = DIRMNRND(N,ALPHA,M) returns M random vectors chosen from the
%   Dirichlet-Multinomial distribution with parameters N and ALPHA.  R is
%   an M-by-K matrix.  Each row of R corresponds to one
%   Dirichlet-Multinomial outcome.
%
%   To generate outcomes from different Dirchlet-Multinomial distributions,
%   ALPHA can also be an M-by-K matrix, where each row contains a different
%   set of parameters.  N can also an M-by-1 vector of positive integers or
%   a positive scalar integer. In this case, dirmnrnd generates each row of
%   R using the corresponding rows of the inputs, or replicates them if
%   needed. If any row of ALPHA are not positive, the corresponding row of
%   R is a 1-by-K vector of NaN values.
%
% Examples
%   See documentation
%
% See also DIRMNFIT, DIRMNPDFLN, DIRMNREG
%
% Copyright 2012-2013 North Carolina State University
% Hua Zhou (hua_zhou@ncsu.edu) and Yiwen Zhang (yzhang31@ncsu.edu)

% check number of arguments
if nargin < 2
    error(message('mglm:dirmnrnd:TooFewInputs'));
end

% If alpha is a column that can be interpreted as a single vector of
% Dir-Mult probabilities, transpose it to a row. Otherwise, treat as a
% matrix with one category. Transpose x if it is a non-empty column vector.
if size(alpha,2)==1 && size(alpha,1)>1
    alpha = alpha';
end

% dimension match
if nargin < 3
    [m,k] = size(alpha);
elseif ~isscalar(m)
    error(message('mglm:dirmnrnd:NonscalarM'));
else
    [mm,k] = size(alpha);
    if ~(mm == 1 || mm == m)
        error('mglm:dirmultrnd:InputSizeMismatch', ...
            'ALPHA must be a row vector or have M rows.');
    end
end
if k < 1
    error(message('mglm:dirmnrnd:NoCategories'));
end
[mm,kk] = size(n);
if kk ~= 1
    error('mglm:dirmnrnd:InputSizeMismatch', ...
        'N must be a scalar, or a column vector with as many rows as ALPHA.');
elseif m == 1 && ~isscalar(n)
    m = mm; % alpha will replicate out to match n
end
if (size(alpha,1)==1)
    alpha = repmat(alpha,m,1);
end

% now generate random vectors
G = randg(alpha);
prob = bsxfun(@times, G, 1./sum(G,2));
ridx = sum(G,2)==0;
if (any(ridx))
    prob(ridx,:) = ...
        mnrnd(1,bsxfun(@times, alpha(ridx,:), 1./sum(alpha(ridx,:),2)));
end
r = mnrnd(n,prob,m);

end

