%% Negative multinomial regression and sparse regression
% A demo of negative multinomial regression and sparse regression

%% Generate negative multinomial random vectors from covariates

clear;
% reset random seed
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
% sample size
n = 200;
% # covariates
p = 15;
% # bins
d = 5;      
% design matrix
X = randn(n,p); 
% true regression coefficients
B = zeros(p,d+1);  
nzidx = [1 3 5];
B(nzidx,:) = ones(length(nzidx),d+1);
eta = X*B;
alpha = exp(eta);
prob(:,d+1) = 1./(sum(alpha(:,1:d),2)+1);
prob(:,1:d) = bsxfun(@times, alpha(:,1:d), prob(:,d+1));
b= binornd(10,0.2, n, 1);
Y = negmnrnd(prob,b);

zerorows = sum(Y,2);
Y=Y(zerorows~=0, :);
X=X(zerorows~=0, :);

%% Fit negative multinomial regression - link over-disperson parameter

tic;
[B_hat, stats] = negmnreg(X,Y);
toc;
display(B_hat);
display(stats);
% Wald test of predictor significance
display('Wald test p-values:');
display(stats.wald_pvalue);

figure;
plot(stats.logL_iter);
xlabel('iteration');
ylabel('log-likelihood');

%% Fit negative multinomial regression - not linking over-disperson parameter

tic;
[B_hat, b_hat, stats] = negmnreg2(X,Y);
toc;
disp(B_hat);
disp(stats.se_B);
disp(b_hat);
disp(stats.se_b);
display(stats);
% Wald test of predictor significance
display('Wald test p-values:');
display(stats.wald_pvalue);

figure;
plot(stats.logL_iter);
xlabel('iteration');
ylabel('log-likelihood');

%% Fit negative multinomial sparse regression - lasso/group/nuclear penalty
% Regression on the over dispersion parameter

penalty = {'sweep','group','nuclear'};
ngridpt = 30;
dist = 'negmn';

for i = 1:length(penalty)
    
    pen = penalty{i};
    [~, stats] = mglm_sparsereg(X,Y,inf,'penalty',pen,'dist',dist);
    maxlambda = stats.maxlambda;

    lambdas = exp(linspace(log(maxlambda),log(maxlambda/100),ngridpt));
    BICs = zeros(1,ngridpt);
    logl =zeros(1, ngridpt);
    dofs = zeros(1, ngridpt);
    tic;
    for j=1:ngridpt
        if j==1
            B0 = zeros(p,d+1);
        else
            B0 = B_hat;
        end
        [B_hat, stats] = mglm_sparsereg(X,Y,lambdas(j),'penalty',pen, ...
            'dist',dist,'B0',B0);
        BICs(j) = stats.BIC;
        logl(j) = stats.logL;
        dofs(j) = stats.dof;
    end
    toc;
    
    % True signal versus estimated signal
    [bestbic,bestidx] = min(BICs);
    [B_best,stats] = mglm_sparsereg(X,Y,lambdas(bestidx),'penalty',pen,'dist',dist);
    figure;
    subplot(1,3,1);
    semilogx(lambdas,BICs);
    ylabel('BIC');
    xlabel('\lambda');
    xlim([min(lambdas) max(lambdas)]);    
    subplot(1,3,2);
    imshow(mat2gray(-B)); title('True B');
    subplot(1,3,3);
    imshow(mat2gray(-B_best)); title([pen ' estimate']);

end

%% Sparse regression (not linking over-disp.) - lasso/group/nuclear penalty
% Do not run regression on the over dispersion parameter

penalty = {'sweep','group','nuclear'};
ngridpt = 30;
dist = 'negmn2';

for i = 1:length(penalty)
    
    pen = penalty{i};
    [~, stats] = mglm_sparsereg(X,Y,inf,'penalty',pen,'dist',dist);
    maxlambda = stats.maxlambda;

    lambdas = exp(linspace(log(maxlambda),log(maxlambda/100),ngridpt));
    BICs = zeros(1,ngridpt);
    logl =zeros(1, ngridpt);
    dofs = zeros(1, ngridpt);
    tic;
    for j=1:ngridpt
        if j==1
            B0 = zeros(p,d);
        else
            B0 = B_hat;
        end
        [B_hat, stats] = mglm_sparsereg(X,Y,lambdas(j),'penalty',pen, ...
            'dist',dist,'B0',B0);
        BICs(j) = stats.BIC;
        logl(j) = stats.logL;
        dofs(j) = stats.dof;
    end
    toc;
    
    % True signal versus estimated signal
    [bestbic,bestidx] = min(BICs);
    B_best = mglm_sparsereg(X,Y,lambdas(bestidx),'penalty',pen,'dist',dist);
    figure;
    subplot(1,3,1);
    semilogx(lambdas,BICs);
    ylabel('BIC');
    xlabel('\lambda');
    xlim([min(lambdas) max(lambdas)]);    
    subplot(1,3,2);
    imshow(mat2gray(-B)); title('True B');
    subplot(1,3,3);
    imshow(mat2gray(-B_best)); title([pen ' estimate']);

end
