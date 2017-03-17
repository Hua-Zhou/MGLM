function [alpha_hat, beta_hat, stats] = gendirmnfit(Y,varargin)
% GENDIRMNFIT Parameter estimates for generalized Dirichlet-Multinomial
%   [ALPHAHAT, BETAHAT] = GENDIRMNFIT(Y) returns maximum likelihood
%   estimates of the parameters of a generalized Dirichlet-Multinomial
%   distribution fit to the count data in Y.
%
%   Input:
%      Y: n-by-d count matrix
%
%   Optional input arguments:
%       'alpha0': (d-1)-by-1 initial parameter value, default is MoM estiamte
%       'beta0': (d-1)-by-1 initial parameter value, default is MoM estiamte
%       'Display': 'off' (default) or 'iter'
%       'MaxIter': maximum iteration, default is 100
%       'TolFun': tolerence in objective value, default is 1e-8
%       'weights': observation weights, default is ones for each obs.
%
%   Output:
%       alpha_hat: d-by-1 parameter estimate
%       beta_hat: d-by-1 parameter estimate
%       stats: a structure holding some estimation statistics
%           AIC: Akaike information criterion
%           BIC: Bayesian information criterion
%           dof: degrees of freedom 2(d-1)
%           gradient: gradient at estiamte
%           iterations: # iterations used
%           logL: log-likelihood at estimate
%           LRT: LRT statistic for comparing to mulitinomial model
%           pvalue: LRT p-value for comparing to mulitinomial model
%           se: d-by-2 standard errors of estimate
%
%
% Examples
%   See documentation
%
% See also GENDIRMNPDFLN, GENDIRMNRND, GENDIRMNREG
%
% Copyright 2015-2017 University of California at Los Angeles
% Hua Zhou (hua_zhou@ncsu.edu)

% parse inputs
argin = inputParser;
argin.addRequired('Y', @isnumeric);
argin.addParamValue('alpha0', [], @(x) isnumeric(x) && all(x>0));
argin.addParamValue('beta0', [], @(x) isnumeric(x) && all(x>0));
argin.addParamValue('Display', 'off', @(x) strcmp(x,'off')||strcmp(x,'iter'));
argin.addParamValue('MaxIter', 100, @(x) isnumeric(x) && x>0);
argin.addParamValue('TolFun', 1e-8, @(x) isnumeric(x) && x>0);
argin.addParamValue('weights', [], @(x) isnumeric(x) && all(x>=0));
argin.parse(Y,varargin{:});

alpha0 = argin.Results.alpha0;
beta0 = argin.Results.beta0;
Display = argin.Results.Display;
MaxIter = argin.Results.MaxIter;
TolFun = argin.Results.TolFun;
wts = argin.Results.weights;

% n=sample size; d=number of categories
[n,d] = size(Y);
if (n<2*(d-1))
    warning('mglm:gendirmnfit:smalln', ...
        'sample size is not large enough for stable estimation');
end
% obtain initial start point from dirithlet-multinomial fit
warning('off','mglm:dirmnfit:badmodel');
dirmn_alpha = dirmnfit(Y);
if ~isempty(alpha0)
    if size(alpha0,1)>1 && size(alpha0,2)==1
        alpha0 = alpha0';
    end
    if size(alpha0,2)~=d-1
        error('mglm:gendirmnfit:alphasize', ...
            'alpha has to be a d-1 dimensional vector');
    end
else
    alpha0 = dirmn_alpha(1:d-1);
end
if ~isempty(beta0)
    if size(beta0,1)>1 && size(beta0,2)==1
        beta0 = beta0';
    end
    if size(beta0,2)~=d-1
        error('mglm:gendirmnfit:betasize', ...
            'beta has to be a d-1 dimensional vector');
    end
else
    beta0 = sum(dirmn_alpha) - cumsum(dirmn_alpha(1:d-1));
end
% set weights
if (isempty(wts))
    wts = ones(n,1);
end

% fit d-1 beta-binomial distribution
Z = bsxfun(@minus, sum(Y,2), [zeros(n,1) cumsum(Y(:,1:end-1),2)]);
alpha_hat = zeros(d-1,1);
beta_hat = zeros(d-1,1);
stats.se = zeros(d-1,2);
iterations = zeros(d-1,1);
stats.gradient = zeros(2, d-1);
for dd=1:d-1
    if isempty(alpha0) || isempty(beta0)
        param0 = [];
    else
        param0 = [alpha0(dd); beta0(dd)];
    end
    [betabin_fit, betabin_stats] = ...
        dirmnfit([Y(:,dd) Z(:,dd+1)],'weights',wts,'MaxIter',MaxIter, ...
        'TolFun',TolFun,'alpha0',param0,'Display',Display);
    alpha_hat(dd) = betabin_fit(1);
    beta_hat(dd) = betabin_fit(2);
    stats.se(dd,:) = betabin_stats.se';
	iterations(dd) = betabin_stats.iterations;
	stats.gradient(:, dd) = betabin_stats.gradient;
end

% output some algorithmic statistics
stats.dof = 2*(d-1);
stats.logL = sum(wts.*gendirmnpdfln(Y,alpha_hat,beta_hat));
stats.BIC = - 2*stats.logL + log(n)*2*(d-1);
stats.AIC = - 2*stats.logL + 2*2*(d-1);
logL_MN = sum(log(mnpdf(Y, sum(Y,1)./sum(Y(:)))));
stats.LRT = 2*(stats.logL - logL_MN);
stats.pvalue = 1 - chi2cdf(stats.LRT, (d-1));
stats.iterations = sum(iterations(dd));

end