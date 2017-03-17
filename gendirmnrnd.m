function [r] = gendirmnrnd(n,alpha,beta,m)
%GENDIRMNRND Random vectors from generalized Dirichlet-Multinomial
%   R = GENDIRMNRND(N,ALPHA,BETA,M) returns a random vector chosen from the
%   generalized Dirichlet-Multinomial distribution with parameters N, ALPHA
%   and BETA.  N is a positive scalar integer specifying the number of
%   trials for each multinomial outcome, also known as the sample size, and
%   ALPHA/BETA is a 1-by-D vector of generalized Dirichlet-Multinomial parameters,
%   where K+1 is the number of bins or categories. Elements of ALPHA/BETA
%   must be positive.  R is a 1-by-D vector containing the counts for each
%   of the D bins. If ALPHA are not positive, R is a 1-by-D vector of NaN
%   values.
%
%   R = GENDIRMNRND(N,ALPHA,BETA,M) returns M random vectors chosen from the
%   Dirichlet-Multinomial distribution with parameters N, ALPHA and BETA.
%   R is an M-by-K matrix.  Each row of R corresponds to one
%   Dirichlet-Multinomial outcome.
%
%   To generate outcomes from different generalized Dirchlet-Multinomial distributions,
%   ALPHA can also be an M-by-D matrix, where each row contains a different
%   set of parameters.  N can also an M-by-1 vector of positive integers or
%   a positive scalar integer. In this case, MNRND generates each row of R
%   using the corresponding rows of the inputs, or replicates them if
%   needed. If any row of ALPHA or BETA are not positive, the corresponding row of
%   R is a 1-by-D vector of NaN values.
%
%   Examples:
%      See documentation
%
%   See also GENDIRMNFIT, GENDIRMNPDFLN, GENDIRMNREG
%
%   Copyright 2012-2013 North Carolina State University 
%   Hua Zhou (hua_zhou@ncsu.edu) and Yiwen Zhang (yzhang31@ncsu.edu)

% check number of arguments
if nargin < 2
    error(message('mglm:gendirmnrnd:TooFewInputs'));
end

% If alpha is a column that can be interpreted as a single vector of
% Dir-Mult probabilities, transpose it to a row. Otherwise, treat as a
% matrix with one category. Transpose x if it is a non-empty column vector.
if size(alpha,2)==1 && size(alpha,1)>1
    alpha = alpha';
end
if size(beta,2)==1 && size(beta,1)>1
    alpha = alpha';
end

% dimension match
if nargin < 4
    [m,k] = size(alpha);
elseif ~isscalar(m)
    error(message('mglm:gendirmnrnd:NonscalarM'));
else
    [mm,k] = size(alpha);
    if ~(mm == 1 || mm == m)
        error('mglm:gendirmnrnd:InputSizeMismatch', ...
            'ALPHA must be a row vector or have M rows.');
    end
end
if k < 1
    error(message('mglm:gendirmnrnd:NoCategories'));
end
[mm,kk] = size(n);
if kk ~= 1
    error('mglm:gendirmnrnd:InputSizeMismatch', ...
        'N must be a scalar, or a column vector with as many rows as ALPHA.');
elseif m == 1 && ~isscalar(n)
    m = mm; % alpha will replicate out to match n
end
if (size(alpha,1)==1)
    alpha = repmat(alpha,m,1);
end
if (size(beta,1)==1)
    beta = repmat(beta,m,1);
end

% now generate random vectors
r = zeros(m,k+1);
r(:,1) = n;
for dd=1:k
    r(:,[dd dd+1]) = dirmnrnd(r(:,dd),[alpha(:,dd) beta(:,dd)],m);
end

end

