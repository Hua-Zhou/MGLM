function [B, stats] = mglm_sparsereg(X,Y,lambda,varargin)
% MGLM_SPARSEREG Sparse regression for multi-response GLM
%   [B,STATS] = MGLM_SPARSEREG(X) returns regularized estimate of the
%   regression parameters of a multi-response MGLM regression with
%   responses Y and covariates X
%
%   Input:
%       X: n-by-p design matrix
%       Y: n-by-d count matrix
%       lambda: penalty constant
%
%   Optional input arguments:
%       'B0': initial parameter value
%       'delta': a constant controlling Nesterov method
%       'Display': 'off' (default) or 'iter'
%       'dist': {'mnlogit'} | 'dirmn' | 'gendirmn' | 'negmn' | 'negmn2'
%       'MaxIter': maximum iteration, default is 100
%       'penalty': penalty type, {'sweep'} | 'group' | 'nuclear'
%       'penidx': p-by-1 logical vector indicating the penalized predictors
%       'pentype': penalty family, {'enet'} | 'power' | 'mcp' | 'scad'
%       'penparam': penalty family parameter, default is 1
%       'TolFun': tolerence in objective value, default is 1e-8
%       'weights': observation weights, default is ones for each obs.
%
%   Output:
%       B: p-by-d parameter estimate
%       stats: a structure holding some estimation statistics
%           AIC: Akaike information criterion
%           BIC: Bayesian information criterion
%           dof: degrees of freedom of the regularized estimate
%           iterations: # iterations used
%           logL: log-likelihood at estimate
%           logL_iter: log-likelihoods at each iteration
%
% Examples
%   See documentation
%
% See also MNLOGITREG, DIRMNREG, GENDIRMNREG, NEGMNREG
%
% Copyright 2015-2017 University of California at Los Angeles
% Hua Zhou (hua_zhou@ncsu.edu)

% parse inputs
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('Y', @isnumeric);
argin.addRequired('lambda', @(x) isnumeric(x) && x>=0);
argin.addParamValue('B0', [], @(x) isnumeric(x));
argin.addParamValue('delta', 1/size(X,1), @(x) isnumeric(x) && x>0);
argin.addParamValue('Display', 'off', ...
    @(x) strcmpi(x,'off')||strcmpi(x,'iter'));
argin.addParamValue('dist', 'mnlogit', @(x) strcmpi(x,'mnlogit') ...
    || strcmpi(x,'dirmn') || strcmpi(x,'negmn') || strcmpi(x,'gendirmn')...
    || strcmpi(x, 'negmn2'));
argin.addParamValue('MaxIter', 100, @(x) isnumeric(x) && x>0);
argin.addParamValue('penalty', 'sweep', @(x) strcmpi(x,'sweep') ...
    || strcmpi(x,'group') || strcmpi(x,'nuclear'));
argin.addParamValue('penidx', [], @(x) islogical(x));
argin.addParamValue('pentype', 'enet', @(x) ischar(x));
argin.addParamValue('penparam', 1, @(x) isnumeric(x));
argin.addParamValue('TolFun', 1e-5, @(x) isnumeric(x) && x>0);
argin.addParamValue('weights', [], @(x) isnumeric(x) && all(x>=0));
argin.addParamValue('overdisp', [], @(x) isnumeric(x) && x>0);
argin.parse(X,Y,lambda,varargin{:});

B0 = argin.Results.B0;
dist = argin.Results.dist;
Display = argin.Results.Display;
MaxIter = argin.Results.MaxIter;
penalty = argin.Results.penalty;
penidx = argin.Results.penidx;
penparam = argin.Results.penparam;
pentype = argin.Results.pentype;
ridgedelta = argin.Results.delta;
TolFun = argin.Results.TolFun;
wts = argin.Results.weights;
overdisp = argin.Results.overdisp;

% n=sample size; d=number of categories
[n,d] = size(Y);
p = size(X,2);
if (size(X,1)~=n)
    error('size of X does not match that of Y');
end
% set weights
if (isempty(wts))
    wts = ones(n,1);
end
% set penalty type
if strcmpi(penalty,'group')
    penalty = 'group_row';
end
% regularization set
if isempty(penidx)
    penidx = true(p,1);
else
    if length(penidx)~=p
        error('size of pendix does not match that of X');
    end
end

% set starting point
if strcmpi(dist,'mnlogit')
    if isempty(B0)
        B0 = zeros(p,d-1);
    else
        if size(B0,1)~=p || size(B0,2)~=d-1
            error('size of B0 does not match model');
        end
    end
elseif strcmpi(dist,'dirmn')
    if isempty(B0)
        B0 = zeros(p,d);
    else
        if size(B0,1)~=p || size(B0,2)~=d
            error('size of B0 does not match model');
        end
    end
elseif strcmpi(dist,'negmn')
    if isempty(B0)
        B0 = zeros(p,d+1);
    else
        if size(B0,1)~=p || size(B0,2)~=d+1
            error('size of B0 does not match model');
        end
    end
elseif strcmpi(dist, 'negmn2')
    if isempty(B0)
        B0 = zeros(p, d);
    else
        if size(B0, 1)~=p|| size(B0,2)~=d;
            error('size of B0 does not match model');
        end
    end
    if isempty(overdisp)
        [~, overdisp, ~] = negmnreg2(X, Y, 'weights', wts);
    end
elseif strcmpi(dist, 'gendirmn')
    if isempty(B0)
        B0 = zeros(p, 2*(d-1));
    else
        if size(B0, 1) ~=p || size(B0, 2) ~= 2*(d-1)
            error('size of B0 does not match model');
        end
    end
end

% batch sizes and pre-allocate arrays
batch_sizes = sum(Y,2);
objval_iter = zeros(1,MaxIter);

% main loop
B = B0;
B_old = B;
alpha_old = 0; alpha = 1;
objval = inf;
isdescent = true;
for iter=1:MaxIter
    % current search point
    if isdescent
        S = B+(alpha_old-1)/alpha*(B-B_old);
    else
        S = B_old+(alpha_old/alpha)*(B-B_old);
    end
    [lossS,lossD1S] = mglmloss(S);
    
    % line search
    B_old = B;
    objval_old = objval;
    for l=1:50
        A = S - ridgedelta*lossD1S;
        B(~penidx,:) = A(~penidx,:);
        [B(penidx,:),penval] = matrix_thresholding(A(penidx,:),...
            ridgedelta*lambda, ...
            'penalty',penalty,'pentype',pentype,'penparam',penparam);
        if nnz(B)==0
            if strcmpi(penalty,'sweep')
                stats.maxlambda = lsq_maxlambda(1,-max(max(A(penidx,:))), ...
                    pentype,penparam)/ridgedelta;
            elseif strcmpi(penalty,'group_row')
                stats.maxlambda = ...
                    max(sqrt(sum(A(penidx,:).^2,2)))/ridgedelta;
            elseif strcmpi(penalty,'nuclear')
                stats.maxlambda = ...
                    lsq_maxlambda(1,-svds(A(penidx,:),1),pentype,penparam) ...
                    /ridgedelta;
            end
            break;
        end
        % objective value
        objval =  mglmloss(B) + penval;
        % surrogate value
        BminusS = B - S;
        surval = lossS + sum(lossD1S(:).*BminusS(:)) ...
            + norm(BminusS,'fro')^2/2/ridgedelta ...
            + penval;
        % line search stopping rule
        if (objval<=surval)
            break;
        else
            ridgedelta = ridgedelta/2;
        end
    end
    
    % force descent
    if (objval<=objval_old) % descent
        % stopping rule
        if (abs(objval_old-objval)<TolFun*(abs(objval_old)+1))
            break;
        end
        isdescent = true;
    else % no descent
        objval = objval_old;
        if isdescent
            isdescent = false;
        else
            break;
        end
    end
    
    % display
    objval_iter(iter) = objval;
    if ~strcmpi(Display,'off')
        display(['iter ' num2str(iter) ...
            ', objval=' num2str(objval_iter(iter))]);
    end
    
    % update alpha constants
    alpha_old = alpha;
    alpha = (1+sqrt(4+alpha_old^2))/2;
end

% output some algorithmic statistics
if strcmpi(penalty,'sweep')
    stats.dof = nnz(~penidx)*size(B,2) + nnz(B(penidx,:));
elseif strcmpi(penalty,'group_row')
    stats.dof = nnz(~penidx)*size(B,2) ...
        + nnz(sum(B(penidx,:).^2,2))*size(B,2);
    stats.dof = nnz(~penidx)*size(B,2)+nnz(sum(B(penidx,:).^2,2))...
        +(size(B,2)-1).*sum(sum(B(penidx,:).^2,2)./sum(A(penidx,:).^2,2));
elseif strcmpi(penalty,'nuclear')
    rank = nnz(abs(svd(B(penidx,:)))>1e-18);
    Aspectrum = svd(A);
    if (p~=d)
        Aspectrum(max(p,d)) = 0;
    end
    stats.dof = 0;
    for i=1:rank
        stats.dof = stats.dof + 1 ...
            + sum(Aspectrum(i)*(Aspectrum(i)-ridgedelta*lambda) ...
            ./(Aspectrum(i)^2-[Aspectrum(1:i-1); Aspectrum(i+1:p)].^2)) ...
            + sum(Aspectrum(i)*(Aspectrum(i)-ridgedelta*lambda) ...
            ./(Aspectrum(i)^2-[Aspectrum(1:i-1); Aspectrum(i+1:d)].^2));
    end
    stats.dof = stats.dof + sum(~penidx)*d;
end

if strcmpi(dist, 'gendirmn')
    stats.dof = stats.dof/2;
end

stats.logL = - mglmloss(B);
stats.AIC = - 2*stats.logL + 2*stats.dof;
stats.BIC = - 2*stats.logL + log(n)*stats.dof;
stats.iterations = iter;

    function [loss,lossD1] = mglmloss(B)
        if strcmpi(dist,'mnlogit')
            P(:,d) = ones(n,1);
            P(:,1:d-1) = exp(X*B);
            P = bsxfun(@times, P, 1./sum(P,2));
            loss = - sum(wts.*mnpdfln(Y,P));
            if nargout<2
                return;
            else
                lossD1 = - reshape(kr({(Y(:,1:d-1) ...
                    - bsxfun(@times,P(:,1:d-1),batch_sizes))',X'})*wts, ...
                    p,d-1);
            end
        elseif strcmpi(dist,'dirmn')
            alpha_matrix = exp(X*B);
            loss = - sum(wts.*dirmnpdfln(Y,alpha_matrix));
            if nargout<2
                return;
            else
                alpha_rowsums = sum(alpha_matrix,2);
                tmpvector = ...
                    psi(alpha_rowsums+batch_sizes)-psi(alpha_rowsums);
                tmpmatrix = psi(alpha_matrix+Y)-psi(alpha_matrix);
                dalpha = bsxfun(@minus, tmpmatrix, tmpvector);
                lossD1 = ...
                    - reshape(kr({(alpha_matrix.*dalpha)',X'})*wts,p,d);
            end
        elseif strcmpi(dist,'gendirmn')
            Z = bsxfun(@minus, sum(Y,2), ...
                [zeros(n,1) cumsum(Y(:,1:end-1),2)]);
            alpha_matrix = exp(X*B);
            a = alpha_matrix(:, 1:(d-1));
            b = alpha_matrix(:, d:(2*(d-1)));
            loss = - sum(wts.*gendirmnpdfln(Y, a, b));
            if nargout < 2
                return;
            else
                dalpha(:,1:d-1) = psi(a+Y(:,1:end-1)) - psi(a) ...
                    - psi(a+b+Z(:,1:(end-1)))+psi(a+b);
                dalpha(:, d:2*(d-1)) = psi(b+Z(:,2:end)) - psi(b) ...
                    - psi(a+b+Z(:,1:(end-1)))+psi(a+b);
                lossD1 = -reshape(kr({(alpha_matrix.*dalpha)', X'})*wts, ...
                    p,2*(d-1));
            end
        elseif strcmpi(dist,'negmn')
            alpha_matrix = exp(X*B);
            alpha_rowsums = sum(alpha_matrix(:,1:d),2)+1;
            P(:,d+1) = 1./alpha_rowsums;
            P(:,1:d) = bsxfun(@times, alpha_matrix(:,1:d), P(:,d+1));
            loss = - sum(wts.*negmnpdfln(Y,P,alpha_matrix(:,d+1)));
            if nargout<2
                return;
            else
                deta = zeros(n,d+1);
                deta(:,1:d) = Y - bsxfun(@times,alpha_matrix(:,1:d) ...
                    ,alpha_matrix(:,d+1)) - bsxfun(@times, P(:,1:d), ...
                    batch_sizes-alpha_matrix(:,d+1).*(alpha_rowsums-1));
                deta(:,d+1) = alpha_matrix(:,d+1) .* ...
                    (psi(alpha_matrix(:,d+1)+batch_sizes) ...
                    - psi(alpha_matrix(:,d+1)) ...
                    + log(P(:,d+1)));
                lossD1 = - reshape(kr({deta',X'})*wts,p,d+1);
            end
        elseif strcmpi(dist, 'negmn2')
            alpha_matrix = exp(X*B);
            alpha_rowsums = sum(alpha_matrix(:,1:d),2)+1;
            P(:,d+1) = 1./alpha_rowsums;
            P(:,1:d) = bsxfun(@times, alpha_matrix(:,1:d), P(:,d+1));
            loss = - sum(wts.*negmnpdfln(Y,P,repmat(overdisp, n, 1)));
            deta = Y - bsxfun(@times,P(:, 1:d),(overdisp+batch_sizes));
            lossD1 = - reshape(kr({deta',X'})*wts,p,d);
        end
    end

end