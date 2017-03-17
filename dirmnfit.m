function [alpha_hat, stats] = dirmnfit(Y,varargin)
% DIRMNFIT Parameter estimates for Dirichlet-Multinomial distribution
%   [PARMHAT, STATS] = DIRMNFIT(Y) returns maximum likelihood estimates of
%   the parameters of a Dirichlet-Multinomial distribution fit to the count
%   data in Y. 
%
%   Input:
%      Y: n-by-d count matrix
%
%   Optional input arguments:
%       'alpha0': d-by-1 initial parameter value, default is MoM estiamte
%       'Display': 'off' (default) or 'iter'
%       'MaxIter': maximum iteration, default is 100
%       'TolFun': tolerence in objective value, default is 1e-8
%       'weights': observation weights, default is ones for each obs.
%
%   Output:
%       alpha_hat: d-by-1 parameter estimate
%       stats: a structure holding some estimation statistics
%           AIC: Akaike information criterion
%           BIC: Bayesian information criterion
%           dof: degrees of freedom
%           gradient: gradient at estiamte
%           iterations: # iterations used
%           logL: log-likelihood at estimate
%           logL_iter: log-likelihoods at each iteration
%           LRT: LRT statistic for comparing to mulitinomial model
%           pvalue: LRT p-value for comparing to mulitinomial model
%           se: d-by-1 standard errors of estimate
%
% Examples
%   See documentation
%
% See also DIRMNPDFLN, DIRMNRND, DIRMNREG
%
% TODO
%   - implement the fast Newton's algorithm derived in ST370
%
% Copyright 2012-2013 North Carolina State University
% Hua Zhou (hua_zhou@ncsu.edu) and Yiwen Zhang (yzhang31@ncsu.edu)

% parse inputs
argin = inputParser;
argin.addRequired('Y', @isnumeric);
argin.addParamValue('alpha0', [], @(x) isnumeric(x) && all(x>0));
argin.addParamValue('Display', 'off', @(x) strcmp(x,'off')||strcmp(x,'iter'));
argin.addParamValue('MaxIter', 100, @(x) isnumeric(x) && x>0);
argin.addParamValue('TolFun', 1e-8, @(x) isnumeric(x) && x>0);
argin.addParamValue('weights', [], @(x) isnumeric(x) && all(x>=0));
argin.parse(Y,varargin{:});

alpha0 = argin.Results.alpha0;
Display = argin.Results.Display;
MaxIter = argin.Results.MaxIter;
TolFun = argin.Results.TolFun;
wts = argin.Results.weights;

% delete observations of batch size zero
zeroidx = sum(Y,2)<eps;
Y(zeroidx,:) = [];
wts(zeroidx) = [];

% n=sample size; d=number of categories
[n,d] = size(Y);
if (n<d)
    warning('mglm:dirmnfit:smalln', ...
        'sample size is not large enough for stable estimation');
end
if (nnz(triu(corr(Y),1)>0)>1)
    warning('mglm:dirmnfit:badmodel', ...
        ['positive correlation between columns; ', ...
        'try generalized DM or negative multinomial']);
end

% set weights
if (isempty(wts))
    wts = ones(n,1);
end

% obtain some summary statistics and auxillary counts
batch_sizes = sum(Y,2);     % batch sizes for each sample pt
nmax = max(batch_sizes);    % maximum batch size
xmax = max(Y(:));           % maximum element of data matrix
% Nk(k) = # sample pts with batch size >=k, or equivalently >k-1
histct = accumarray(batch_sizes, wts);
Nk = n - [0; cumsum(histct(1:end-1))];
% S(j,k) = # sample pts with j-th coord >=k
S = zeros(d,xmax);
for j=1:d
    histct = accumarray(Y(:,j)+1, wts);
    if (length(histct)<xmax+1)
        histct(xmax+1) = 0;
    end
    S(j,:) = n - cumsum(histct(1:end-1,:))';
end

% set starting point if not provided
if (isempty(alpha0))
    pi0 = sum(bsxfun(@times, Y, wts),1)';
    pi0 = pi0/sum(pi0);
    freq = bsxfun(@times, Y, 1./sum(Y,2));
    rho = sum(sum(bsxfun(@times, freq.^2, wts),1) ...
        ./sum(bsxfun(@times,freq,wts),1));
    alpha0 = pi0*(d-rho)/(rho-1);
end

% pre-compute the constant term in log-likelihood
logL_term = sum(gammaln(batch_sizes+1)) - sum(sum(gammaln(Y+1)));
logL_iter = zeros(1,MaxIter);
logL_iter(1) = obj_fctn(alpha0);
if (strcmpi(Display,'iter'))
    disp(['iterate = 1', ' logL = ', num2str(logL_iter(1))]);
end

% main loop
alpha_hat = alpha0;
for iter=2:MaxIter
    alpha_MM = mm_update(alpha_hat);
    obj_MM = obj_fctn(alpha_MM);
    alpha_Newton = newton_update(alpha_hat);
    alpha_Newton(alpha_Newton<=0) = alpha_hat(alpha_Newton<=0)/2;
    obj_Newton = obj_fctn(alpha_Newton);
    if (obj_MM>=obj_Newton)
        alpha_hat = alpha_MM;
        logL_iter(iter) = obj_MM;
    else
        alpha_hat = alpha_Newton;
        logL_iter(iter) = obj_Newton;
    end
    % display
    if (strcmpi(Display,'iter'))
        disp(['iterate = ', num2str(iter), ...
            ' logL = ', num2str(logL_iter(iter))]);
    end    
    % termination criterion
    if (abs(logL_iter(iter)-logL_iter(iter-1)) ...
            < TolFun*(abs(logL_iter(iter))+1))
        break;
    end
end

% Check the gradient is 0 or not. 
tmpb = S./bsxfun(@plus, alpha_hat, 0:xmax-1);
grad = sum(tmpb,2) - sum(Nk./(sum(alpha_hat)+(0:nmax-1)'));
if( mean(grad.^2) >1e-4)
    disp(['The algorithm does not converge within ',  ...
        num2str(iter), ' iteration']);
end

% output some algorithmic statistics
stats.BIC = - 2*logL_iter(iter) + log(n)*d;
stats.AIC = - 2*logL_iter(iter) + 2*d;
stats.dof = d;
stats.iterations = iter;
stats.logL = logL_iter(iter);
stats.logL_iter = logL_iter(1:iter);
b = 1./sum(S./bsxfun(@plus, alpha_hat, 0:xmax-1).^2,2);
a = sum(Nk./(sum(alpha_hat)+(0:nmax-1)').^2);
stats.se = sqrt(b+a/(1-a*sum(b))*b.^2);
stats.gradient = grad;
logL_MN = sum(log(mnpdf(Y, sum(Y,1)./sum(Y(:)))));
stats.LRT = 2*(stats.logL - logL_MN);
stats.pvalue = 1 - chi2cdf(stats.LRT, 1);

    function fval = obj_fctn(param)
        fval = sum(sum(S.*log(bsxfun(@plus, param, 0:xmax-1)))) ...
            - sum(Nk.*log(sum(param)+(0:(nmax-1))')) + logL_term;
    end

    function new_param = mm_update(old_param)
        new_param = sum(S./bsxfun(@plus, old_param, 0:xmax-1), 2) ...
            / sum(Nk./(sum(old_param)+(0:nmax-1)')) .* old_param;
    end

    function new_param = newton_update(old_param)
        tmpb = S./bsxfun(@plus, old_param, 0:xmax-1);
        grad = sum(tmpb,2) - sum(Nk./(sum(old_param)+(0:nmax-1)'));
        tmpb = 1./sum(tmpb./bsxfun(@plus, old_param, 0:xmax-1),2);
        tmpa = sum(Nk./(sum(old_param)+(0:nmax-1)').^2);
        new_param = old_param + grad.*tmpb + ...
            tmpa*sum(grad.*tmpb)/(1-tmpa*sum(tmpb))*tmpb;
    end

end