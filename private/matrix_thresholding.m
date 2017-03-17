function [B,penalty_value] = matrix_thresholding(X,lambda,varargin)
% MATRIX_THRESHOLDING Thresholding a matrix M
%   [B] = MATRIX_THESHOLDING(X,lambda) returns the minimizer of
%   0.5*||X-B||_F^2 + lambda*pen(B)
%
% COPYRIGHT North Carolina State University 
% AUTHOR: Hua Zhou (hua_zhou@ncsu.edu) and Yiwen Zhang

% parse inputs
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('lambda', @(x) isnumeric(x) && x>=0);
argin.addParamValue('penalty', 'sweep', @(x) strcmpi(x,'sweep') || ...
    strcmpi(x,'group_row') || strcmpi(x,'group_col') || strcmpi(x,'nuclear'));
argin.addParamValue('pentype', 'enet', @(x) ischar(x));
argin.addParamValue('penparam', 1, @(x) isnumeric(x));
argin.parse(X,lambda,varargin{:});

lambda = argin.Results.lambda;
penalty = argin.Results.penalty; 
pentype = argin.Results.pentype; 
penparam = argin.Results.penparam;

% n=sample size; d=number of categories
[m,n] = size(X);

% thesholding
B = zeros(m,n);
if strcmpi(penalty,'sweep')
    B(:) = lsq_thresholding(ones(m*n,1),-X(:),lambda,pentype,penparam);
    penalty_value = lambda*sum(penalty_function(B(:),lambda,pentype,penparam));
elseif strcmpi(penalty,'group_row')
    row_l2norm = sqrt(sum(X.^2,2));
    B = bsxfun(@times, X, max(1-lambda./row_l2norm,0));
    penalty_value = lambda*sum(sqrt(sum(B.^2,2)));
elseif strcmpi(penalty,'group_col')
    col_l2norm = sqrt(sum(X.^2,1));
    B = bsxfun(@times, X, max(1-lambda./col_l2norm,0));
    penalty_value = lambda*sum(sqrt(sum(B.^2,1)));
elseif strcmpi(penalty,'nuclear')
    [U,s,V] = svt(X,lambda,'pentype',pentype,'penparam',penparam);
    if isempty(s)
        B = zeros(m,n);
    else
        B = U*bsxfun(@times,s,V');
    end
    penalty_value = lambda*sum(penalty_function(s,lambda,pentype,penparam));
end

end