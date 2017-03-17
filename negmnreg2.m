function [B,b,stats] = negmnreg2(X,Y,varargin)
% NEGMNREG2 Parameter estimates for negative multinomial regression
%   [B,b,STATS] = NEGMNREG(X) returns maximum likelihood estimates of the
%   regression parameters of a negative multinomial regression with
%   responses Y and covariates X, *without* linking over-dispersion
%   parameter to covariates
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
%       b: over-dispersion parameter estimate
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
% EXAMPLEs
%   See documentation
%
% See also NEGMNREG, NEGMNPDFLN, NEGMNFIT, NEGMNRND
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
[n, d] = size(Y);
p = size(X, 2);
if (size(X, 1)~=n)
    error('size of X does not match that of Y');
end
if (n<p*d)
    warning('mglm:negmnreg2:smalln', ...
        'sample size is not large enough for stable estimation');
end
% set weights
if (isempty(wts))
    wts = ones(n,1);
end

% turn off warnings
warning('off','stats:glmfit:IterationLimit');
warning('off','stats:glmfit:BadScaling');
warning('off','stats:glmfit:IllConditioned');

% set starting point
if(isempty(B0))
	B0 = zeros(p, d);
	for i =1:d
		B0(:,i) = glmfit_priv(X, Y(:,i), 'poisson', ...
			'weights', wts, 'constant', 'off');
	end
	A = exp(X*B0);
	P(:,d+1) = 1./(sum(A,2)+1);
    P(:,1:d) = bsxfun(@times, A, P(:,d+1));
else 
	if size(B0,1)~=p || size(B0, 2)~= d;
		error('mglm:negmnreg2:B0', ...
			'size of B0 should be p-by-d');
	end
	A = exp(X*B0);
	P(:,d+1) = 1./(sum(A,2)+1);
    P(:,1:d) = bsxfun(@times, A, P(:,d+1));
end

[p0,b0] = negmnfit(Y); %#ok<ASGLU>
D = repmat(b0, n, 1);

% pre-compute the constant term in log-likelihood
batch_sizes = sum(Y,2);
logL_iter = zeros(1,MaxIter);
logL_iter(1) = sum(wts.*negmnpdfln(Y,P,D));

if (strcmpi(Display,'iter'))
    disp(['iterate = 0', ' logL = ', num2str(logL_iter(1))]);
end

% main loop
B = B0;
b = b0;

for iter = 1:MaxIter
	[B, b, A, LL] = param_update(A, B, b);
	logL_iter(iter+1) = LL;
	
	% display 
	if (strcmp(Display, 'iter'))
		disp(['iter = ', num2str(iter), ...
			' logL= ', num2str(logL_iter(iter))]);
	end
	% termination criterion
	if ((iter>1) && (abs(logL_iter(iter) - logL_iter(iter -1))...
		< TolFun*(abs(logL_iter(iter))+1)))
		break;
	end
end

% turn on warnings;
warning on all;

% output some algorithmic statistics
stats.BIC = - 2*logL_iter(iter) + log(n)*p*d;
stats.BIC = - 2*logL_iter(iter) + log(n)*p*d;
stats.iterations = iter;
stats.logL = logL_iter(iter);
stats.logL_iter = logL_iter(1:iter);

P(:,d+1) = 1./(sum(A,2)+1);
P(:,1:d) = bsxfun(@times, A, P(:,d+1));
tmpv1 = psi(b+batch_sizes) - psi(b);
% check diverge 
stats.se = nan(p,(d+1));
stats.wald_stat = nan(1,p);
stats.wald_pvalue = nan(1,p);
stats.H = nan(p*(d+1), p*(d+1));

if any(isnan(A(:))) || any(isinf(A(:))) || any(isnan(b)) || any(isnan(b))
    warning('mglm:negmnreg2:diverge',...
        ['Regression parameters diverge. '...
        'No SE or test results reported. '...
        'Recommend multinomial logit model']);
else 
	% calculate dl
	dlbeta = sum( tmpv1 - log(sum(A,2)+1));
	deta = Y - bsxfun(@times, P(:, 1:d), (b+batch_sizes));
 	score = bsxfun(@times, kr({deta',X'})', wts);
 	score = sum(score, 1);
 	score = [score, dlbeta]';
	stats.gradient = score;
	% calculate H
    hbeta = sum( psi(1, b+batch_sizes) - psi(1, b));
	upleft = kr({P(:, 1:d)', X'});
	upright = -sum(upleft, 2);
	upleft = upleft*bsxfun(@times, upleft', wts.*(b+batch_sizes));
	
	for dd=1:d
		idx = (dd-1)*p + (1:p);
 		upleft(idx, idx) = upleft(idx, idx) - ...
			X'*bsxfun(@times, X, wts.*(b+batch_sizes).*P(:,dd));
	end
    up = [upleft, upright];
    bottom = [upright', hbeta];
	H = [ up; bottom ];
	stats.H = H;
	stats.observed_information = -H;
	
	% check dl
	if mean(score.^2) >1e-4
		warning('mglm:negmnreg2:notcnvg',...
            'The algorithm does not converge.  Please check gradients.');
        disp(score);
	end 
	% check H
	Heig = eig(H);
    if any(Heig>0)
        warning('mglm:negmnreg2:Hnotpd', ...
           ['Hessian at estimate not pos. def.. ' ...
           'No standard error estimates or test results are given']);
    elseif any(Heig == 0 )
        warning('mglm:negmnreg2:Hsig', ...
            ['Hessian at estimate is almost singular. '...
            'No standard error estimates or test results are given']);
    elseif all(Heig < 0)       
        Hinv = inv(H);
        SE = sqrt(-diag(Hinv));
		stats.se_B = reshape(SE(1:(p*d)), p, d);
		stats.se_b = SE(p*d+1);
        stats.wald_stat = zeros(1,p);
        stats.wald_pvalue = zeros(1,p);
        for pp=1:p
            idx = pp:p:(pp+p*(d-1));
            stats.wald_stat(pp) = -B(idx)*(Hinv(idx,idx)\B(idx)');
            stats.wald_pvalue(pp) = 1 - chi2cdf(stats.wald_stat(pp),d);
        end
    end
  
end   
	
	function [B, b, A, LL, score] = param_update(old_A, old_B, old_b)
		% obtain distribution parameter
		alpha_rowsums = sum(old_A,2)+1;
		P(:,d+1) = 1./alpha_rowsums;
		P(:,1:d) = bsxfun(@times, old_A, P(:,d+1));
		tmpBeta = psi(old_b +batch_sizes) - psi(old_b);
		
		% MM update
		dlbeta = sum( tmpBeta - log(alpha_rowsums) );
		hbeta = sum( psi(1, old_b+batch_sizes) - psi(1, old_b));
		b_MM = old_b;
		if(hbeta<0) 
			tempb_MM = old_b - dlbeta/hbeta;
            if( tempb_MM > 0) 
                b_MM = tempb_MM; 
            end
		end
		
		w_B = (b_MM + batch_sizes)./alpha_rowsums;
		wky = bsxfun(@times, Y, wts./w_B);
		B_MM = zeros(p, d);
        
        for ii=1:d;
			B_MM(:,ii) = glmfit_priv(X,wky(:,ii),'poisson', ...
                'weights',w_B,'constant','off','b0',old_B(:,ii));            
        end
        
        A_MM = exp(X*B_MM);
		P_MM(:,d+1) = 1./(sum(A_MM,2)+1);
		P_MM(:,1:d) = bsxfun(@times, A_MM, P_MM(:,d+1));
        D_MM = repmat(b_MM, n, 1);
		LL_MM = sum(wts.*negmnpdfln(Y, P_MM, D_MM));
        
		% Newton update
		deta = Y - bsxfun(@times, P(:, 1:d), (old_b + batch_sizes) );
		score = bsxfun(@times, kr({deta',X'})', wts);
		score = sum(score, 1);
		score = [score, dlbeta]';
		
		upleft = kr({P(:, 1:d)', X'});
		upright = -sum(upleft, 2);
		upleft = upleft*bsxfun(@times,upleft',(wts.*(old_b+batch_sizes)));
		
        for kk=1:d
			idx = (kk-1)*p + (1:p);
 			upleft(idx, idx) = upleft(idx, idx) - ...
 				X'*bsxfun(@times, X, wts.*(old_b+batch_sizes).*P(:,kk));
        end
        up = [upleft, upright];
        bottom = [upright', hbeta];
		hessian = [ up; bottom ];
        
        Heig = eig(hessian);
        if any(Heig>=0)
            LL_Newton = nan;
        else 
		Newton_temp = hessian\score;
		B_Newton = old_B - reshape(Newton_temp(1:(p*d)), p , d);
		b_Newton = old_b - Newton_temp(end);
		A_Newton = exp(X*B_Newton);
        P_Newton(:,d+1) = 1./(sum(A_Newton(:,1:d),2)+1);
        P_Newton(:,1:d) = bsxfun(@times,A_Newton(:,1:d),P_Newton(:,d+1));
        D_Newton = repmat(b_Newton, n, 1);
		LL_Newton = sum(wts.*negmnpdfln(Y,P_Newton,D_Newton)); 
			
            % Half stepping
            if (b_Newton>0 || ~isnan(LL_Newton) || LL_Newton >=0) ...
                    && (LL_Newton < LL_MM)
				llnewiter = nan(1, 5);
				llnewiter(1) = LL_Newton;
                for step =1:5
					Newton_temp = (hessian\score)./(2^step);
					B_N = old_B - reshape(Newton_temp(1:(p*d)), p, d);
					b_N = old_b - Newton_temp(end);
					A_N = exp(X*B_N);
                    P_N = [A_N(:,1:d) ones(size(X,1), 1)];
                    P_N = bsxfun(@times, P_N, 1./sum(P_N,2));
					D_N = repmat(b_N, n, 1);
					llnew = sum(wts.*negmnpdfln(Y,P_N,D_N));
                    if (llnew < llnewiter(step))
                        break;
                    else
                        llnewiter(step+1)=llnew;
                        B_Newton = B_N;
                        b_Newton = b_N;
                        A_Newton = A_N;
                        LL_Newton = llnew;
                    end
                end
            end
        end
        % Pick the optimal update
        if LL_Newton>0 || isnan(LL_Newton) || LL_MM >= LL_Newton
            A = A_MM;
            B = B_MM;
            b = b_MM;
            LL = LL_MM; 
        else
            A = A_Newton;
            B = B_Newton;
            b = b_Newton;
            LL = LL_Newton;
        end
    end
end
