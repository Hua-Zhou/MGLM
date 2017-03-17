function [r] = negmnrnd(prob,b,m)
%NEGMNRND Random vectors from the negative multinomial distribution.
%   R = NEGMNRND(PROB,B) returns a random vector chosen from the negative
%   multinomial distribution with parameters PROB and B.  PROB is a K+1
%   probability vector, and B is a shape parameter. R is a 1-by-K vector
%   containing the counts for each of the K bins. If PROB is not a
%   probability vector, R is a 1-by-K vector of NaN values.
%
%   R = NEGMNRND(PROB,B,M) returns M random vectors chosen from the
%   negative multinomial distribution with parameters PROB and M.  R is an
%   M-by-K matrix.  Each row of R corresponds to one negative multinomial
%   outcome.
%
%   To generate outcomes from different negative multinomial distributions,
%   PROB can also be an M-by-(K+1) matrix, where each row contains a
%   different set of parameters.  N can also an M-by-1 vector of positive
%   integers or a positive scalar integer. In this case, MNRND generates
%   each row of R using the corresponding rows of the inputs, or replicates
%   them if needed. If any row of PROB are not a probability vector, the
%   corresponding row of R is a 1-by-K vector of NaN values.
%
%   Examples:
%    See documentation
%
%   See also NEGMNFIT, NEGMNPDFLN, NEGMNREG, NEGMNREG2
%
%   Copyright 2012-2013 North Carolina State University
%   Hua Zhou (hua_zhou@ncsu.edu) and Yiwen Zhang (yzhang31@ncsu.edu)

% check number of arguments
if nargin < 2
    error(message('mglm:negmnrnd:TooFewInputs'));
end

% If PROB is a column that can be interpreted as a single vector of
% Dir-Mult probabilities, transpose it to a row. Otherwise, treat as a
% matrix with one category. Transpose x if it is a non-empty column vector.
if size(prob,2)==1 && size(prob,1)>1
    prob = prob';
end

% dimension match
if nargin < 3
    [m,k] = size(prob);
elseif ~isscalar(m)
    error(message('mglm:negmnrnd:NonscalarM'));
else
    [mm,k] = size(prob);
    if ~(mm == 1 || mm == m)
        error('mglm:negmnrnd:InputSizeMismatch', ...
            'ALPHA must be a row vector or have M rows.');
    end
end
if k < 1
    error(message('mglm:negmnrnd:NoCategories'));
end
[mm,kk] = size(b);
if kk ~= 1
    error('mglm:negmnrnd:InputSizeMismatch', ...
        'N must be a scalar, or a column vector with as many rows as ALPHA.');
elseif m == 1 && ~isscalar(b)
    m = mm; % alpha will replicate out to match n
end
if (size(prob,1)==1)
    prob = repmat(prob,m,1);
end

% now generate random vectors
G = gamrnd(b,1./prob(:,end)-1);
r = poissrnd(bsxfun(@times, prob(:,1:end-1), G./(1-prob(:,end))));

end

