function [p_hat,b_hat,stats] = negmnfit(Y,varargin)
% NEGMNFIT Parameter estimates for negative multinomial distribution
%   [P_HAT,B_HAT,STATS] = NEGMNFIT(X) returns maximum likelihood estimates
%   of the parameters of a negative multinomial distribution fit to the
%   count data in Y.
%
%   Input:
%      Y: n-by-d count matrix
%
%   Optional input arguments:
%       'p0': (d+1)-by-1 initial value for the prob parameter
%       'b0': initial value for the over-dispersion parameter
%       'Display': 'off' (default) or 'iter'
%       'MaxIter': maximum iteration, default is 100
%       'TolFun': tolerence in objective value, default is 1e-8
%       'weights': observation weights, default is ones for each obs.
%
%   Output:
%       p_hat: (d+1)-by-1 parameter estimate
%       b_hat: over-dispersion estimate
%       stats: a structure holding some estimation statistics
%           AIC: Akaike information criterion
%           BIC: Bayesian information criterion
%           dof: degrees of freedom
%           gradient: gradient at estiamte
%           iterations: # iterations used
%           logL: log-likelihood at estimate
%           logL_iter: log-likelihoods at each iteration
%           se: (d+1)-by-1 standard errors of estimate
%
% Examples
%   See documentation
%
% TODO
%   - implement half stepping
%
% Copyright 2015-2017 University of California at Los Angeles
% Hua Zhou (hua_zhou@ncsu.edu)

% parse inputs
argin = inputParser;
argin.addRequired('Y', @isnumeric);
argin.addParamValue('p0', [], @(x) isnumeric(x) && all(x>0) && abs(sum(x)-1)<1e-6);
argin.addParamValue('b0', [], @(x) isscalar(x));
argin.addParamValue('Display', 'off', @(x) strcmp(x,'off')||strcmp(x,'iter'));
argin.addParamValue('MaxIter', 100, @(x) isnumeric(x) && x>0);
argin.addParamValue('TolFun', 1e-8, @(x) isnumeric(x) && x>0);
argin.addParamValue('weights', [], @(x) isnumeric(x) && all(x>=0));
argin.parse(Y,varargin{:});

b0 = argin.Results.b0;
Display = argin.Results.Display;
MaxIter = argin.Results.MaxIter;
p0 = argin.Results.p0;
TolFun = argin.Results.TolFun;
wts = argin.Results.weights;

% n=sample size; d=number of categories
[n,d] = size(Y);
if (n<d+1)
    warning('mglm:negmnfit:smalln', ...
        'sample size is not large enough for stable estimation');
end
if (nnz(triu(corr(Y),1)<0)>0)
    warning('mglm:negmnfit:badmodel', ...
        ['negative correlation between columns; ', ...
        'try Dirichlet-Multinomial or generalized Dirichlet-Multinomial']);
end
% set weights
if (isempty(wts))
    wts = ones(n,1);
end

% obtain some summary statistics and auxillary counts
batch_sizes = sum(Y,2);     % batch sizes for each sample pt
bin_sizes = sum(bsxfun(@times, Y, wts),1);  % bin sizes for each bin/category
nmax = max(batch_sizes);    % maximum batch size
% Nk(k) = # sample pts with batch size >=k, or equivalently >k-1
histct = accumarray(batch_sizes+1, wts);
Nk = n - cumsum(histct(1:end-1));

% set starting point if either p0 or b0 is not provided
if (isempty(p0) || isempty(b0))
    bsmean = sum(wts.*batch_sizes)/sum(wts);
    bsvar = sum(wts.*(batch_sizes-bsmean).^2)/sum(wts);
    p0 = zeros(1,d+1);
    if (bsmean<bsvar)
        b0 = bsmean^2/(bsvar-bsmean);
        p0(d+1) = bsmean/bsvar;
        p0(1:d) = p0(d+1)/b0*bin_sizes/sum(wts);
    else
        b0 = 1e6;
        p0(1:d) = 1e-6;
        p0(d+1) = 1-sum(p0(1:d));
    end
else
    if (length(p0)~=d+1 || any(p0<=0) || abs(sum(p0)-1)>1e-6)
        error('mglm:negmnfit:prob0dim', ...
            'p0 has to be a probability vector of length d+1');
    end
    if (~isscalar(b0) || b0<=0)
        error('mglm:negmnfit:b0positive', ...
            'b0 has to be a positive scalar');
    end
end

% pre-compute the constant term in log-likelihood
logL_term = - sum(sum(gammaln(Y+1)));
logL_iter = zeros(1,MaxIter);
logL_iter(1) = obj_fctn(p0,b0);
if (strcmpi(Display,'iter'))
    disp(['iterate = 1', ' logL = ', num2str(logL_iter(1))]);
end

% main loop
b_hat = b0; p_hat = p0;
for iter=2:MaxIter
    [p_MM,b_MM] = mm_update(p_hat,b_hat);
    obj_MM = obj_fctn(p_MM,b_MM);
    [p_Newton,b_Newton] = newton_update(p_hat,b_hat);
    obj_Newton = obj_fctn(p_Newton,b_Newton);
    if (obj_MM>=obj_Newton)
        p_hat = p_MM; b_hat = b_MM;
        logL_iter(iter) = obj_MM;
    else
        p_hat = p_Newton; b_hat = b_Newton;
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

% Check score is 0 or not
score = [bin_sizes./p_hat(1:d)-sum(wts)*b_hat/p_hat(d+1), ...
     sum(Nk./(b_hat+(0:nmax-1)'))+sum(wts)*log(p_hat(d+1))]';
if( mean(score.^2) >1e-4)
    disp(['The algorithm does not converge within ',  num2str(iter),...
        ' iteration'])
end

% output some algorithmic statistics
stats.BIC = - 2*logL_iter(iter) + log(n)*(d+1);
stats.AIC = - 2*logL_iter(iter) + 2*2*(d-1);
stats.iterations = iter;
stats.gradient = score;
stats.logL = logL_iter(iter);
stats.logL_iter = logL_iter(1:iter);
% compute standard errors
dinv = [p_hat(1:d).^2./bin_sizes, ...
    1/(sum(Nk./(b_hat+(0:nmax-1)').^2)-sum(wts)/b_hat)]';
r1v = repmat(sqrt(b_hat*sum(wts))/p_hat(d+1),d+1,1);
r1v(end) = sqrt(sum(wts)/b_hat);
stats.se = sqrt(dinv-(dinv.*r1v).^2/(1+sum(dinv.*r1v.^2)));
stats.dof = d+1;

    function fval = obj_fctn(prob,b)
        fval = sum(Nk.*log(b+(0:nmax-1)')) + sum(bin_sizes.*log(prob(1:d))) ...
            + sum(wts)*b*log(prob(d+1)) + logL_term;
    end

    function [new_p,new_b] = mm_update(old_p,old_b)
        new_b = - old_b*sum(Nk./(old_b+(0:nmax-1)')) ...
            /sum(wts)/log(old_p(d+1));
        new_p = [bin_sizes sum(wts)*new_b];
        new_p = new_p/sum(new_p);
    end

    function [new_p,new_b,score] = newton_update(old_p,old_b)
        score = [bin_sizes./old_p(1:d)-sum(wts)*old_b/old_p(d+1), ...
            sum(Nk./(old_b+(0:nmax-1)'))+sum(wts)*log(old_p(d+1))]';
        diaginv = [old_p(1:d).^2./bin_sizes, ...
            1/(sum(Nk./(old_b+(0:nmax-1)').^2)-sum(wts)/old_b)]';
        rank1v = repmat(sqrt(old_b*sum(wts))/old_p(d+1),d+1,1);
        rank1v(end) = sqrt(sum(wts)/old_b);
        newton_iterate = [old_p(1:d), old_b]' + diaginv.*score ...
            - sum(diaginv.*rank1v.*score)/(1+sum(diaginv.*rank1v.^2)) ...
            *(diaginv.*rank1v);
        new_p(1:d) = newton_iterate(1:d)';
        new_p(d+1) = 1 - sum(new_p(1:d));
        if any(new_p<=0)
            new_p = old_p;
        end
        if (newton_iterate(end)>0)
            new_b = newton_iterate(end);
        else
            new_b = old_b;
        end
    end

end