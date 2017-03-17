function [B, stats] = mnlogitreg(X,Y,varargin)
% MNLOGITREG Parameter estimates for multinomial-logit regression
%   B = MNLOGITREG(X) returns maximum likelihood estimates of the
%   regression parameters of a multinomial logit regression with
%   responses Y and covariates X
%
%   Input:
%       X: n-by-p design matrix
%       Y: n-by-d count matrix
%
%   Optional input arguments:
%       'B0': p-by-(d-1) initial parameter value
%       'Display': 'off' (default) or 'iter'
%       'MaxIter': maximum iteration, default is 100
%       'TolFun': tolerence in objective value, default is 1e-8
%       'weights': observation weights, default is ones for each obs.
%
%   Output:
%       B: p-by-(d-1) parameter estimate
%       stats: a structure holding some estimation statistics
%           AIC: Akaike information criterion
%           BIC: Bayesian information criterion
%           dof: degrees of freedom p(d-1)
%           gradient: p(d-1)-by-1 gradient at estimate
%           H: p(d-1)-by-p(d-1) Hessian at estimate
%           iterations: # iterations used
%           logL: log-likelihood at estimate
%           logL_iter: log-likelihoods at each iteration
%           observed_information: pd-by-pd obs. info. matrix at estimate
%           prob: n-by-d fitted probabilities
%           se: p-by-(d-1) standard errors of estimate
%           wald_stat: p-by-1 Wald statistics for testing predictor effects
%           wald_pvalue: p-by-1 Wald p-values for testing predictor effects
%           yhat: n-by-d fitted values
%
% Examples
%   See documentation
%
% See also
%
% Copyright 2015-2017 University of California at Los Angeles
% Hua Zhou (hua_zhou@ncsu.edu)

% parse inputs
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('Y', @isnumeric);
argin.addParamValue('B0', [], @(x) isnumeric(x));
argin.addParamValue('Display', 'off', @(x) strcmp(x,'off')||strcmp(x,'iter'));
argin.addParamValue('MaxIter', 100, @(x) isnumeric(x) && x>0);
argin.addParamValue('TolFun', 1e-8, @(x) isnumeric(x) && x>0);
argin.addParamValue('weights', [], @(x) isnumeric(x) && all(x>=0));
argin.parse(X,Y,varargin{:});

B0 = argin.Results.B0;
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
if (n<p*(d-1))
    warning('mnlogitreg:smalln', ...
        'sample size is not large enough for stable estimation');
end
% set weights
if (isempty(wts))
    wts = ones(n,1);
end

% set starting point
if (isempty(B0))
    B0 = zeros(p,d-1);
    P = sum(bsxfun(@times, Y, wts),1)/sum(wts.*sum(Y,2));
    P = repmat(P,n,1);
else
    P(:,d) = ones(n,1);
    P(:,1:d-1) = exp(X*B0);
    P = bsxfun(@times, P, 1./sum(P,2));
end

% pre-compute the constant term in log-likelihood
batch_sizes = sum(Y,2);
logL_iter = zeros(1,MaxIter);
logL_iter(1) = sum(wts.*mnpdfln(Y,P));
if (strcmpi(Display,'iter'))
    disp(['iterate = 1', ' logL = ', num2str(logL_iter(1))]);
end

% main loop
B = B0;
P_MM = zeros(n,d);
P_Newton = zeros(n,d);
for iter=1:MaxIter
    [B, P, LL] = param_update(B,P);
	logL_iter(iter) = LL;
    % display
    if (strcmpi(Display,'iter'))
        disp(['iterate = ', num2str(iter), ...
            ' logL = ', num2str(logL_iter(iter))]);
    end    
    % termination criterion
    if ((iter>1) && (abs(logL_iter(iter)-logL_iter(iter-1)) ...
            < TolFun*(abs(logL_iter(iter))+1)))
        break;
    end
end

% output some algorithmic statistics
stats.BIC = - 2*logL_iter(iter) + log(n)*p*(d-1);
stats.AIC = - 2*logL_iter(iter) + 2*p*(d-1);
stats.dof = p*(d-1);
stats.iterations = iter;
stats.logL = logL_iter(iter);
stats.logL_iter = logL_iter(1:iter);
stats.prob = P;
stats.yhat = bsxfun(@times, stats.prob, sum(Y,2));

% compute gradient
dl = kr({(Y(:,1:d-1)-bsxfun(@times,P(:,1:d-1),batch_sizes))',X'})*wts;
stats.gradient = dl;
% compute Hessian matrix
H = kr({P(:,1:d-1)',X'});
H = bsxfun(@times,H,(wts.*batch_sizes)')*H';
for i=1:d-1
    idx = (i-1)*p + (1:p);
    H(idx,idx) = H(idx,idx) - X'*bsxfun(@times,X,wts.*batch_sizes.*P(:,i));
end
stats.observed_information = -H;
stats.H = H;

% compute standard errors
stats.se = Inf(p, d-1);
stats.wald_stat = NaN(1,p);
stats.wald_pvalue=NaN(1,p);

% check convergence
if mean(dl.^2) > 1e-4
    warning('mglm:mnlogitreg:notcnvg', ...
        'The algorithm does not converge.  Please check gradient.');
end
% check saddle point
Heig = eig(H);
if any(Heig>0)
    warning('mglm:mnlogitreg:Hnotpd', ...
        ['Hessian at estimate not pos. def.. '...
        'No standard error estimates or test results are given']);
	elseif any(Heig == 0 )
    warning('mglm:mnlogitreg:Hsig', ...
        ['Hessian at estimate is almost singular. '...
        'No standard error estimates or test results are given']);
	else
    Hinv = inv(H);
    stats.se = reshape(sqrt(-diag(Hinv)),p,d-1);
    stats.wald_stat = zeros(1,p);
    stats.wald_pvalue = zeros(1,p);
        for pp=1:p
            idx = pp:p:(pp+p*(d-2));
            stats.wald_stat(pp) = -B(idx)*(Hinv(idx,idx)\B(idx)');
            stats.wald_pvalue(pp) = 1 - chi2cdf(stats.wald_stat(pp),d-1);
        end
end

    function [B, P, LL] = param_update(old_B,old_P)
        % MM update
        wkwts = batch_sizes.*old_P(:,d);
        wky = bsxfun(@times, Y, 1./wkwts);
        wkwts = wts.*wkwts;
        B_MM = zeros(p,d-1);
        for dd=1:d-1
            B_MM(:,dd) = glmfit_priv(X,wky(:,dd),'poisson', ...
                'weights',wkwts,'constant','off','b0',old_B(:,dd));            
        end
        if (nargout<2)
            return;
        end
		
		P_MM(:,d) = 1;
		P_MM(:,1:d-1) = exp(X*B_MM);
		P_MM = bsxfun(@times, P_MM, 1./sum(P_MM,2));
		LL_MM = sum(wts.*mnpdfln(Y,P_MM));
			
        % Newton update
        score = ...
            kr({(Y(:,1:d-1)-bsxfun(@times,old_P(:,1:d-1),batch_sizes))',X'})*wts;
        hessian = kr({old_P(:,1:d-1)',X'});
        hessian = bsxfun(@times,hessian,(wts.*batch_sizes)')*hessian';
        for dd=1:d-1
            idx = (dd-1)*p + (1:p);
            hessian(idx,idx) = hessian(idx,idx) - X' * bsxfun(@times, X, ...
                wts.*batch_sizes.*old_P(:,dd));
        end
        
        Heig = eig(hessian);
        if any(Heig >=0)
            LL_Newton = nan;
        else 
            B_Newton = old_B - reshape(hessian\score, p, d-1);
            P_Newton(:,d) = 1;
            P_Newton(:,1:d-1) = exp(X*B_Newton);
            P_Newton = bsxfun(@times, P_Newton, 1./sum(P_Newton,2));
            LL_Newton = sum(wts.*mnpdfln(Y,P_Newton));

            % Half stepping		
            if ( ~isnan(LL_Newton)||LL_Newton >=0 ) && (LL_MM > LL_Newton) 
                llnewiter = nan(1,5);
                llnewiter(1) = LL_Newton;
                for step=1:5
                    B_N = old_B - reshape(hessian\score, p, d-1)./(2^step);
                    P_N = [exp(X*B_N) ones(size(X,1),1)];
                    P_N = bsxfun(@times, P_N, 1./sum(P_N,2));
                    llnew = sum(wts.*mnpdfln(Y,P_N));
                    if llnew < llnewiter(step) 
                        break; 
                    else 
                        llnewiter(step+1)= llnew; 
                        B_Newton = B_N;
                        P_Newton = P_N;
                        LL_Newton = llnew;
                    end
                end
            end	
        end
        
		% Compare the two method and give the optimal update
        if ( LL_Newton>0 || isnan(LL_Newton) || LL_MM>=LL_Newton)
			B = B_MM;
			P = P_MM;
			LL= LL_MM;
			else
			B = B_Newton;
			P = P_Newton;
			LL = LL_Newton;
        end
    end
end