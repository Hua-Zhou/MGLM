function [B1,B2,stats] = gendirmnreg(X,Y,varargin)
% GENDIRMNREG Generalized Dirichlet-Multinomial regression
%   [B1,B2] = GENDIRMNREG(X) returns maximum likelihood estimates of the
%   regression parameters of a generalized Dirichlet-Multinomial regression
%   with responses Y and covariates X
%
%   Input:
%       X: n-by-p design matrix
%       Y: n-by-d count matrix
%
%   Optional input arguments:
%       'B10': p-by-(d-1) initial parameter value
%       'B20': p-by-(d-1) initial parameter value
%       'Display': 'off' (default) or 'iter'
%       'MaxIter': maximum iteration, default is 100
%       'TolFun': tolerence in objective value, default is 1e-8
%       'weights': observation weights, default is ones for each obs.
%
%   Output:
%       B1: p-by-(d-1) parameter estimate
%       B2: p-by-(d-1) parameter estimate
%       stats: a structure holding some estimation statistics
%           AIC: Akaike information criterion
%           BIC: Bayesian information criterion
%           dof: degrees of freedom 2p(d-1)
%           gradient: p-by-2(d-1) gradient at estimate
%           H: 2p(d-1)-by-2p(d-1) Hessian at estimate
%           iterations: # iterations used
%           logL: log-likelihood at estimate
%           observed_information: 2p(d-1)-by-2p(d-1) obs. info. matrix
%           se: p-by-2(d-1) standard errors of estimate
%           wald_stat: p-by-1 Wald statistics for testing predictor effects
%           wald_pvalue: p-by-1 Wald p-values for testing predictor effects
%           yhat: n-by-d fitted values
%
% Examples
%   See documentation
%
% See also GENDIRMNPDFLN, GENDIRMNRND, GENDIRMNFIT
%
% COPYRIGHT (2012-2013) North Carolina State University 
% Hua Zhou (hua_zhou@ncsu.edu) and Yiwen Zhang (yzhang31@ncsu.edu)

% parse inputs
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('Y', @isnumeric);
argin.addParamValue('B10', [], @(x) isnumeric(x));
argin.addParamValue('B20', [], @(x) isnumeric(x));
argin.addParamValue('Display', 'off', @(x) strcmp(x,'off')||strcmp(x,'iter'));
argin.addParamValue('MaxIter', 100, @(x) isnumeric(x) && x>0);
argin.addParamValue('TolFun', 1e-8, @(x) isnumeric(x) && x>0);
argin.addParamValue('weights', [], @(x) isnumeric(x) && all(x>=0));
argin.parse(X,Y,varargin{:});

B10 = argin.Results.B10;
B20 = argin.Results.B20;
Display = argin.Results.Display;
MaxIter = argin.Results.MaxIter;
TolFun = argin.Results.TolFun;
wts = argin.Results.weights;

% n=sample size; d=number of categories
[n,d] = size(Y);
p = size(X,2);
if (size(X,1)~=n)
    error('size of X does not match that of Y');
end
if (n<2*p*(d-1))
    warning('mglm:gendirmnreg:smalln', ...
        'sample size is not large enough for stable estimation');
end
% set weights
if (isempty(wts))
    wts = ones(n,1);
end

% set starting point from dirichlet-multinomial regression fit
if isempty(B10) || isempty(B20)
    dirmn_B = dirmnreg(X,Y);
    if isempty(B10)
        B10 = dirmn_B(:,1:d-1);
    end
    if isempty(B20)
        dirmn_A = exp(X*dirmn_B);
        B20 = X\log(bsxfun(@minus,sum(dirmn_A,2),cumsum(dirmn_A(:,1:d-1),2)));
    end
end

% fit d-1 beta-binomial regressions
Z = bsxfun(@minus, sum(Y,2), [zeros(n,1) cumsum(Y(:,1:end-1),2)]);
B1 = zeros(p,d-1);
B2 = zeros(p,d-1);
dl = zeros(2*p,d-1);
iterations=zeros(1, d-1);
for dd=1:d-1
    [betabinreg_fit, betabinreg_stats] = ...
        dirmnreg(X,[Y(:,dd) Z(:,dd+1)], 'weights', wts, 'Display', Display,...
        'B0', [B10(:,dd) B20(:,dd)], 'MaxIter', MaxIter, 'TolFun', TolFun);
    B1(:,dd) = betabinreg_fit(:,1);
    B2(:,dd) = betabinreg_fit(:,2);
    dl(:,dd) = betabinreg_stats.gradient;
    iterations(dd) = betabinreg_stats.iterations;
end

dl = [dl(1:p,:) dl((p+1):(2*p),:)];
stats.gradient = dl;
   
% check convergence
if mean(dl.^2)>1
    warning('mglm:gendirmnreg:notcnvg',...
        'The algorithm does not converge.  Please check gradient.');
end

% estimate the Standard deviation

stats.se = inf(p, 2*(d-1));
stats.wald_stat = nan(1, p);
stats.wald_pvalue = nan(1,p);


eb1 = exp(X*B1);
eb2 = exp(X*B2);
tmpmatrix1 = psi(0, eb1+Y(:,1:(d-1))) - psi(0, eb1);
tmpmatrix2 =-psi(1, eb1+Y(:,1:(d-1))) + psi(1, eb1);
a1 = eb1.*tmpmatrix1- (eb1.^2).*tmpmatrix2;

tmpmatrix1 = psi(0, eb1+eb2+Z(:,1:(d-1))) - psi(0, eb1+eb2);
tmpmatrix2 =-psi(1, eb1+eb2+Z(:,1:(d-1))) + psi(1, eb1+eb2);
a2 = tmpmatrix1.*eb1 - tmpmatrix2.*eb1.^2;

b = eb1.*eb2.*(-psi(1, eb1+eb2+Z(:,1:(d-1)))+psi(1, eb1+eb2));
d1= eb2.*(psi(0, eb2+Z(:,2:d))-psi(0, eb2)) -...
    eb2.^2.*(-psi(1, eb2+Z(:,2:d))+psi(1, eb2));
d2= eb2.*(psi(0, eb1+eb2+Z(:,1:(d-1)))-psi(0,eb1+eb2))-...
    eb2.^2.*(-psi(1, eb1+eb2+Z(:,1:(d-1)))+psi(1, eb1+eb2));

Ha = zeros(p*(d-1), p*(d-1));
for dd=1:(d-1)
    idx = (dd-1)*p+ (1:p);
    Ha(idx, idx) = X'*bsxfun(@times, wts.*(a1(:,dd)-a2(:,dd)), X);
end

Hb = zeros(p*(d-1), p*(d-1));
for dd=1:(d-1)
    idx = (dd-1)*p + (1:p);
    Hb(idx, idx) = X'*bsxfun(@times, wts.*b(:,dd), X);
end
Hd = zeros(p*(d-1), p*(d-1));
for dd=1:(d-1)
    idx = (dd-1)*p + (1:p);
    Hd(idx, idx) = X'*bsxfun(@times, wts.*(d1(:,dd)-d2(:,dd)), X);
end

H1= [Ha Hb];
H2= [Hb' Hd];
H = [H1;H2];
stats.H = H;

stats.wald_stat = nan(1, p);
stats.wald_pvalue = nan(1, p);

if( sum(sum(isnan(H))) > 0 )
	warning('mglm:gendirmreg:Hnan', ...
		['Out of range of psi function. '...
        'No standard error estimate or test results reported.']);
	stats.se = nan(p, 2*(d-1));
	else
		Heig = eig(H);
		if any(Heig>0)
		warning('mglm:gendirmnreg:Hnotpd', ...
			['Hessian at estimate not pos. def.. '
            'No standard error estimate or test results reported.']);
		elseif any(Heig == 0 )
		warning('mglm:gendirmnreg:Hsig', ...
			['Hessian at estimate is almost singular. ', ...
            'No standard error estimate or test results reported.']);
		elseif all(Heig <0);        
		Hinv = inv(-H);
		stats.se = reshape(sqrt(diag(Hinv)), p, 2*(d-1));
		stats.observed_information = -H;
			for pp=1:p
				idx = pp + p*(0:(2*(d-1)-1));
				stats.wald_stat(pp) = ...
                    [B1(pp,:) B2(pp,:)]*(Hinv(idx, idx)\[B1(pp,:) ...
                    B2(pp,:)]');
				stats.wald_pvalue(pp)=1-chi2cdf(stats.wald_stat(pp), d);
			end
		end
end


% output some algorithmic statistics
stats.logL = sum(wts.*gendirmnpdfln(Y,exp(X*B1),exp(X*B2)));
stats.BIC = - 2*stats.logL + log(n)*2*p*(d-1);
stats.AIC = - 2*stats.logL + 2*2*p*(d-1);
stats.dof = 2*p*(d-1);
stats.iterations = sum(iterations);

end