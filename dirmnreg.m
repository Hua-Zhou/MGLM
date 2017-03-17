function [B, stats] = dirmnreg(X,Y,varargin)
% DIRMNREG Dirichlet-Multinomial regression
%   [B, STATS] = DIRMNREG(X,Y) returns maximum likelihood estimates of the
%   regression parameters of a Dirichlet-Multinomial regression with
%   responses Y and covariates X. 
%
%   Input:
%       X: n-by-p design matrix
%       Y: n-by-d count matrix
%
%   Optional input arguments:
%       'B0': p-by-d initial parameter value
%       'Display': 'off' (default) or 'iter'
%       'MaxIter': maximum iteration, default is 100
%       'TolFun': tolerence in objective value, default is 1e-8
%       'weights': observation weights, default is ones for each obs.
%
%   Output:
%       B: p-by-d parameter estimate
%       stats: a structure holding some estimation statistics
%           AIC: Akaike information criterion
%           BIC: Bayesian information criterion
%           dof: degrees of freedom
%           gradient: pd-by-1 gradient at estimate
%           H: pd-by-pd Hessian at estimate
%           iterations: # iterations used
%           logL: log-likelihood at estimate
%           logL_iter: log-likelihoods at each iteration
%           observed_information: pd-by-pd obs. info. matrix at estimate
%           se: p-by-d standard errors of estimate
%           wald_stat: p-by-1 Wald statistics for testing predictor effects
%           wald_pvalue: p-by-1 Wald p-values for testing predictor effects
%           yhat: n-by-d fitted values
%
% Examples
%   See documentation
%
% See also DIRMNFIT, DIRMNPDFLN, DIRMNRND
%
% Copyright 2012-2013 North Carolina State University 
% Hua Zhou (hua_zhou@ncsu.edu) and Yiwen Zhang (yzhang31@ncsu.edu)

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
if (n<p*d)
    warning('mglm:dirmnreg:smalln', ...
        'sample size is not large enough for stable estimation');
end
% set weights
if (isempty(wts))
    wts = ones(n,1);
end

% set starting point
if (isempty(B0))
    B0 = zeros(p,d);
    warning('off','mglm:dirmnfit:badmodel');
    A = repmat(dirmnfit(Y,'weights',wts)',n,1);
else
    A = exp(X*B0);
end

% pre-compute the constant term in log-likelihood
batch_sizes = sum(Y,2);
logL_iter = zeros(1,MaxIter);
logL_iter(1) = sum(wts.*dirmnpdfln(Y,A));
if (strcmpi(Display,'iter'))
    disp(['iterate = 0', ' logL = ', num2str(logL_iter(1))]);
end

% turn off warnings
warning('off','stats:glmfit:IterationLimit');
warning('off','stats:glmfit:BadScaling');
warning('off','stats:glmfit:IllConditioned');

% main loop
B = B0;
for iter=1:MaxIter
    [B, A, logL_iter(iter)] = param_update(B,A);
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

% turn on warnings
warning on all;

% output some algorithmic statistics
stats.BIC = - 2*logL_iter(iter) + log(n)*p*d;
stats.AIC = - 2*logL_iter(iter) + 2*p*d;
stats.dof = p*d;
stats.iterations = iter;
stats.logL = logL_iter(iter);
stats.logL_iter = logL_iter(1:iter);
stats.yhat = bsxfun(@times, A, sum(Y,2)./sum(A,2));

tmpv = psi(sum(A,2)+sum(Y,2))-psi(sum(A,2));
tmpm = psi(A+Y)-psi(A);
tmpv2 = -(psi(1,sum(A,2)+sum(Y,2))-psi(1,sum(A,2)));
tmpm2 = -(psi(1,A+Y)-psi(1,A));
tmpm2 = A.*tmpm - A.^2.*tmpm2;

% check diverge 
stats.se = nan(p,d);
stats.wald_stat = nan(1,p);
stats.wald_pvalue = nan(1,p);
stats.H = nan(p*d, p*d);

if any(isnan(A(:))) || any(isinf(A(:))) || any(isnan(tmpv)) ...
        || any(isnan(tmpm(:))) 
    warning('mglm:dirmnreg:diverge',...
        ['Regression parameters diverge. ', ...
        'No SE or test results reported. ', ...
        'Recommend multinomial logit model']);
else
    % calculate gradients
    dl = bsxfun(@minus, tmpm, tmpv);
    dl = kr({(A.*dl)', X'})*wts;
    stats.gradient = dl;
    % calculate Hessian
    H = kr({A',X'});
    H = bsxfun(@times,H,(tmpv2.*wts)')*H';
    for ddd=1:size(Y,2)
        idxx = (ddd-1)*size(X,2) + (1:size(X,2));
        H(idxx,idxx) = H(idxx,idxx) + X' * bsxfun(@times, X, ...
            wts.*(-tmpv.*A(:,ddd)+tmpm2(:,ddd)));
    end
    stats.observed_information = -H;

    % compute standard errors
    % First check convergence
    if mean(dl.^2)> 1e-2
        warning('mglm:dirmnreg:notcnvg',...
            ['The algorithm does not converge. ', ...
            'Please check gradient.']);
    else 
        Heig = eig(H);
        if any(Heig>0)
        warning('mglm:dirmnreg:Hnotpd', ...
            ['Hessian at estimate not pos. def.. ', ...
            'No standard error estimates or test results are given']);
        elseif any(Heig == 0 )
        warning('mglm:dirmnreg:Hsig', ...
            ['Hessian at estimate is almost singular. ', ...
            'No standard error estimates or test results are given']);
        else        
        Hinv = inv(H);
        stats.se = reshape(sqrt(-diag(Hinv)),p,d);
        % Wald test of covariates
        stats.wald_stat = zeros(1,p);
        stats.wald_pvalue = zeros(1,p);
            for pp=1:p
                idx = pp:p:(pp+p*(d-1));
                stats.wald_stat(pp) = -B(idx)*(Hinv(idx,idx)\B(idx)');
                stats.wald_pvalue(pp) = 1 - chi2cdf(stats.wald_stat(pp),d);
            end
        end
    end
end
    


    function [B, A, LL] = param_update(old_B,old_A)

        % row sums of data y and alpha matrix
        alpha_rowsums = sum(old_A,2);
        tmpvector = psi(alpha_rowsums+batch_sizes)-psi(alpha_rowsums);
        tmpmatrix = psi(old_A+Y)-psi(old_A);

        % MM update
        wkwts = wts.*tmpvector;
        wky = old_A.*tmpmatrix;
        wky = bsxfun(@times, wky, 1./tmpvector);
        B_MM = zeros(size(X,2),size(Y,2));
        for dd=1:d
            B_MM(:,dd) = glmfit_priv(X,wky(:,dd),'poisson', ...
                'weights',wkwts,'constant','off','b0',old_B(:,dd));
        end
        A_MM = exp(X*B_MM);
        LL_MM = sum(wts.*dirmnpdfln(Y,A_MM));

        % Newton update
        dalpha = bsxfun(@minus, tmpmatrix, tmpvector);
        score = kr({(old_A.*dalpha)',X'})*wts;
        tmpvector2 = -(psi(1,alpha_rowsums+batch_sizes)-psi(1,alpha_rowsums));
        tmpmatrix2 = -(psi(1,old_A+Y)-psi(1,old_A));
        tmpmatrix2 = old_A.*(tmpmatrix - old_A.*tmpmatrix2);
        hessian = kr({old_A',X'});
        hessian = bsxfun(@times,hessian,(tmpvector2.*wts)')*hessian';
        for dd=1:d
            idx = (dd-1)*p + (1:p);
            hessian(idx,idx) = hessian(idx,idx) + X' * bsxfun(@times, X, ...
                wts.*(-tmpvector.*old_A(:,dd)+tmpmatrix2(:,dd)));
        end
        Heig = eig(hessian);
        if any(Heig >=0)
            LL_Newton = nan;
        else 
            B_Newton = old_B - reshape(hessian\score, p, d);
            A_Newton = exp(X*B_Newton);
            LL_Newton = sum(wts.*dirmnpdfln(Y,A_Newton));

            % Half stepping
            if ( ~isnan(LL_Newton)||LL_Newton >=0 ) && (LL_MM > LL_Newton) 
                llnewiter = nan(1,5);
                llnewiter(1) = LL_Newton;
                for step = 1:5
                    B_N = old_B - reshape(hessian\score, p, d)./(2^step);
                    A_N = exp(X*B_N);
                    llnew = sum(wts.*dirmnpdfln(Y, A_N));
                    if llnew < llnewiter(step)
                        break;
                    else 
                        llnewiter(step+1) = llnew;
                        B_Newton = B_N;
                        A_Newton = A_N;
                        LL_Newton = llnew;
                    end
                end
            end
        end
		
		% Comapre the two methods and choose the better update
		if (isnan(LL_Newton) || LL_MM>=LL_Newton)
			B = B_MM;
			A = A_MM;
			LL = LL_MM;
        else
			B = B_Newton;
			A = A_Newton;
			LL = LL_Newton;
		end
    end
end