function [B, stats] = negmnreg(X,Y,varargin)
% NEGMNREG Parameter estimates for negative multinomial regression
%   [B,STATS] = NEGMNREG(X) returns maximum likelihood estimates of the
%   regression parameters of a negative multinomial regression with
%   responses Y and covariates X
%
%   Input:
%       X: n-by-p design matrix
%       Y: n-by-d count matrix
%
%   Optional input arguments:
%       'B0': p-by-(d+1) initial parameter value
%       'Display': 'off' (default) or 'iter'
%       'MaxIter': maximum iteration, default is 100
%       'TolFun': tolerence in objective value, default is 1e-8
%       'weights': observation weights, default is ones for each obs.
%
%   Output:
%       B: p-by-(d+1) parameter estimate
%       stats: a structure holding some estimation statistics
%           AIC: Akaike information criterion
%           BIC: Bayesian information criterion
%           dof: degrees of freedom p(d+1)
%           gradient: p(d+1)-by-1 gradient at estimate
%           H: p(d+1)-by-p(d+1) Hessian at estimate
%           iterations: # iterations used
%           logL: log-likelihood at estimate
%           logL_iter: log-likelihoods at each iteration
%           observed_information: p(d+1)-by-p(d+1) obs. info. matrix
%           se: p-by-d standard errors of estimate
%           wald_stat: 1-by-p Wald statistics for testing predictor effects
%           wald_pvalue: 1-by-p Wald p-values for testing predictor effects
%
% COPYRIGHT 2012-2013 North Carolina State University
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
    warning('mglm:negmnreg:smalln', ...
        'sample size is not large enough for stable estimation');
end
% set weights
if (isempty(wts))
    wts = ones(n,1);
end

% turn off glmfit warnings
warning('off','stats:glmfit:IterationLimit');
warning('off','stats:glmfit:BadScaling');
warning('off','stats:glmfit:IllConditioned');

% set starting point
if (isempty(B0))
    B0 = zeros(p,d+1);
    for i=1:d
        B0(:,i) = glmfit_priv(X,Y(:,i),'poisson', ...
            'weights',wts,'constant','off');
    end
    B0(:,d+1) = glmfit_priv(X,sum(Y,2)+1,'poisson', ...
        'weights',wts,'constant','off','estdisp','on');
    A = exp(X*B0);
    P(:,d+1) = 1./(sum(A(:,1:d),2)+1);
    P(:,1:d) = bsxfun(@times, A(:,1:d), P(:,d+1));
    D = A(:,d+1);
else
    if size(B0,1)~=p || size(B0,2)~=d+1
        error('mglm:negmnreg:B0', ...
            'size of B0 should be p-by-(d+1)');
    end
    A = exp(X*B0);
    P(:,d+1) = 1./(sum(A(:,1:d),2)+1);
    P(:,1:d) = bsxfun(@times, A(:,1:d), P(:,d+1));
    D = A(:,d+1);
end

% pre-compute the constant term in log-likelihood
batch_sizes = sum(Y,2);
logL_iter = zeros(1,MaxIter);
logL_iter(1) = sum(wts.*negmnpdfln(Y,P,D));
if (strcmpi(Display,'iter'))
    disp(['iterate = 0', ' logL = ', num2str(logL_iter(1))]);
end

% main loop
B = B0;
P_MM = zeros(n,d+1);
P_Newton = zeros(n,d+1);
for iter=1:MaxIter
    [B, A, LL] = param_update(B,A);
    logL_iter(iter+1) = LL;
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
stats.BIC = - 2*logL_iter(iter) + log(n)*p*(d+1);
stats.dof = p*(d+1);
stats.iterations = iter;
stats.logL = logL_iter(iter);
stats.logL_iter = logL_iter(1:iter);

alpha_rowsums = sum(A(:,1:d),2)+1;
prob(:,d+1) = 1./alpha_rowsums;
prob(:,1:d) = bsxfun(@times, A(:,1:d), prob(:,d+1));
tmpv2 = A(:,d+1)+sum(Y,2);
tmpv1 = psi(tmpv2)-psi(A(:,d+1));
% check diverge
stats.se = nan(p,(d+1));
stats.wald_stat = nan(1,p);
stats.wald_pvalue = nan(1,p);
stats.H = nan(p*(d+1), p*(d+1));

if any(isnan(A(:))) || any(isinf(A(:))) || any(isnan(tmpv2)) ...
        || any(isnan(tmpv1))
    warning('mglm:negmnreg:diverge',...
        ['Regression parameters diverge. '...
        'No SE or test results reported. '...
        'Recommend multinomial logit model']);
else
    % calculate dl
    deta = zeros(n,d+1);
    deta(:,1:d) = Y - bsxfun(@times, A(:,1:d),A(:,d+1)) ...
        - bsxfun(@times, prob(:,1:d), ...
        batch_sizes- A(:,d+1).*(alpha_rowsums-1));
    deta(:,d+1) = A(:,d+1) .* ...
        (psi(A(:,d+1)+batch_sizes)-psi(A(:,d+1)) ...
        +log(prob(:,d+1)));
    score = kr({deta',X'})*wts;
    stats.gradient = score;
    % calculate H
    H = kr({[prob(:,1:d) -A(:,d+1)./tmpv2]', X'});
    H = bsxfun(@times,H,(wts.*tmpv2)') * H';
    for i=1:d
        idx = (i-1)*p + (1:p);
        H(idx,idx) = H(idx,idx) - X' * bsxfun(@times, X, ...
            wts.*tmpv2.*prob(:,i));
    end
    idx = d*p+(1:p);
    tmpv2 = A(:,d+1).* ...
        (tmpv1 ...
        + A(:,d+1).*(psi(1,tmpv2)-psi(1,A(:,d+1))) ...
        + log(prob(:,d+1)) - A(:,d+1)./tmpv2);
    H(idx,idx) = H(idx,idx) + X'*bsxfun(@times,X,wts.*tmpv2);
    stats.H = H;
    stats.observed_information = -H;
    % check dl
    if mean(score.^2) >1e-4
        warning('mglm:negmnreg:notcnvg',...
            'The algorithm does not converge.  Please check gradient.');
        disp(score);
    end
    
    % check H
    Heig = eig(H);
    if any(Heig>0)
        warning('mglm:negmnreg:Hnotpd', ...
            ['Hessian at estimate not pos. def.. '...
            'No standard error estimates or test results are given']);
    elseif any(Heig == 0 )
        warning('mglm:negmnreg:Hsig', ...
            ['Hessian at estimate is almost singular. '...
            'No standard error estimates or test results are given']);
    elseif all(Heig < 0)
        Hinv = inv(H);
        stats.se = reshape(sqrt(-diag(Hinv)),p,d+1);
        stats.wald_stat = zeros(1,p);
        stats.wald_pvalue = zeros(1,p);
        for pp=1:p
            idx = pp:p:(pp+p*d);
            stats.wald_stat(pp) = -B(idx)*(Hinv(idx,idx)\B(idx)');
            stats.wald_pvalue(pp) = 1 - chi2cdf(stats.wald_stat(pp),d+1);
        end
    end
    
end

    function [B, A, LL] = param_update(old_B,old_alpha)
        % obtain distribution parameter
        alpha_rowsums = sum(old_alpha(:,1:d),2)+1;
        tmpvector1 = psi(old_alpha(:,d+1)+batch_sizes)-psi(old_alpha(:,d+1));
        % MM update of the shape parameter regression coefficients
        B_MM = zeros(p,d+1);
        wkwts = log(alpha_rowsums);
        dlbeta = sum( bsxfun(@times, X, ...
            (tmpvector1-wkwts).*old_alpha(:,d+1) ), 1);
        dlbeta = reshape(dlbeta, p, 1);
        hbeta_w = (psi(1, old_alpha(:,d+1)+batch_sizes) ...
            - psi(1, old_alpha(:,d+1)) +...
            tmpvector1 - wkwts ).*old_alpha(:, d+1);
        hbeta = bsxfun(@times, kr({X', X'})', hbeta_w);
        hbeta = reshape(sum(hbeta,1), p, p);
        
        if( ~any(eig(hbeta)>0) )
            B_MM(:, d+1) = old_B(:, d+1) - reshape(hbeta\dlbeta, p, 1);
        else
            wky = old_alpha(:,d+1).*tmpvector1;
            wky = wky./wkwts;
            wkwts = wts.*wkwts;
            B_MM(:,d+1) = glmfit_priv(X,wky,'poisson', ...
                'weights',wkwts,'constant','off','b0',old_B(:,d+1));
        end
        % MM update of the regular regression coefficients
        wkwts = (exp(X*B_MM(:,d+1))+batch_sizes)./alpha_rowsums;
        wky = bsxfun(@times, Y, 1./wkwts);
        wkwts = wts.*wkwts;
        for dd=1:d
            B_MM(:,dd) = glmfit_priv(X,wky(:,dd),'poisson', ...
                'weights',wkwts,'constant','off','b0',old_B(:,dd));
        end
        if (nargout<2)
            return;
        end
        
        A_MM = exp(X*B_MM);
        P_MM(:,d+1) = 1./(sum(A_MM(:,1:d),2)+1);
        P_MM(:,1:d) = bsxfun(@times, A_MM(:,1:d), P_MM(:,d+1));
        
        LL_MM = sum(wts.*negmnpdfln(Y,P_MM,A_MM(:,d+1)));
        
        % Newton update
        prob(:,d+1) = 1./alpha_rowsums;
        prob(:,1:d) = bsxfun(@times, old_alpha(:,1:d), prob(:,d+1));
        deta = zeros(n,d+1);
        deta(:,1:d) = Y - bsxfun(@times,old_alpha(:,1:d),old_alpha(:,d+1)) ...
            - bsxfun(@times, prob(:,1:d), ...
            batch_sizes-old_alpha(:,d+1).*(alpha_rowsums-1));
        deta(:,d+1) = old_alpha(:,d+1) .* ...
            (psi(old_alpha(:,d+1)+batch_sizes)-psi(old_alpha(:,d+1)) ...
            +log(prob(:,d+1)));
        score = kr({deta',X'})*wts;
        tmpvector2 = old_alpha(:,d+1)+batch_sizes;
        hessian = kr({[prob(:,1:d) -old_alpha(:,d+1)./tmpvector2]', X'});
        hessian = bsxfun(@times,hessian,(wts.*tmpvector2)') * hessian';
        for dd=1:d
            idx = (dd-1)*p + (1:p);
            hessian(idx,idx) = hessian(idx,idx) - X' * bsxfun(@times, X, ...
                wts.*tmpvector2.*prob(:,dd));
        end
        idx = d*p+(1:p);
        tmpvector2 = old_alpha(:,d+1).*(tmpvector1 + ...
            old_alpha(:,d+1).*(psi(1,tmpvector2)-psi(1,old_alpha(:,d+1))) ...
            + log(prob(:,d+1)) - old_alpha(:,d+1)./tmpvector2);
        hessian(idx,idx) = hessian(idx,idx) ...
            + X'*bsxfun(@times,X,wts.*tmpvector2);
        if( any(eig(hessian) > 0) )
            LL_Newton = nan;
        else
            B_Newton = old_B - reshape(hessian\score, p, d+1);
            A_Newton = exp(X*B_Newton);
            P_Newton(:,d+1) = 1./(sum(A_Newton(:,1:d),2)+1);
            P_Newton(:,1:d) = bsxfun(@times,A_Newton(:,1:d),P_Newton(:,d+1));
            LL_Newton = sum(wts.*negmnpdfln(Y,P_Newton,A_Newton(:,d+1)));
            
            % Half stepping
            if( ~isnan(LL_Newton)||LL_Newton >=0)&&(LL_Newton < LL_MM)
                llnewiter = nan(1, 5);
                llnewiter(1) = LL_Newton;
                for step = 1:5
                    B_N = old_B - reshape(hessian\score, p, d+1)./(2^step);
                    A_N = exp(X*B_N);
                    P_N = [A_N(:,1:d) ones(size(X,1),1)];
                    P_N = bsxfun(@times, P_N, 1./sum(P_N,2));
                    llnew = sum(wts.*negmnpdfln(Y,P_N,A_N(:,d+1)));
                    if llnew < llnewiter(step)
                        break;
                    else
                        llnewiter(step+1)=llnew;
                        B_Newton = B_N;
                        A_Newton = A_N;
                        LL_Newton = llnew;
                    end
                end
            end
        end
        % Pick the optimal update
        if (isnan(LL_Newton) || LL_MM >= LL_Newton)
            A = A_MM;
            B = B_MM;
            LL = LL_MM;
        else
            A = A_Newton;
            B = B_Newton;
            LL = LL_Newton;
        end
    end

end