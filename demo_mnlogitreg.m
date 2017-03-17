%% Multinomial-logit regression and sparse regression
% A demo of Multinomial-logit regression and sparse regression

%% Generate multinomial random vectors from covariates

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
% true regression coefficients: predictors 1, 3, and 5 have effects
B = zeros(p,d-1);
nzidx = [1 3 5];
B(nzidx,:) = ones(length(nzidx),d-1);
prob = zeros(n,d);
prob(:,d) = ones(n,1);
prob(:,1:d-1) = exp(X*B);
prob = bsxfun(@times, prob, 1./sum(prob,2));
batchsize = 25+unidrnd(25,n,1);
Y = mnrnd(batchsize,prob);

zerorows = sum(Y,2);
Y=Y(zerorows~=0, :);
X=X(zerorows~=0, :);

%% Fit multinomial logit regression

tic;
[B_hat, stats] = mnlogitreg(X,Y);
toc;
display(B_hat);
display(stats.se);
display(stats);
% Wald test of predictor significance
display('Wald test p-values:');
display(stats.wald_pvalue);

figure;
plot(stats.logL_iter);
xlabel('iteration');
ylabel('log-likelihood');

%% Fit multinomial logit sparse regression - - lasso/group/nuclear penalty

penalty = {'sweep','group','nuclear'};
ngridpt = 10;
dist = 'mnlogit';

for i = 1:length(penalty)
    
    pen = penalty{i};
    [~, stats] = mglm_sparsereg(X,Y,inf,'penalty',pen,'dist',dist);
    maxlambda = stats.maxlambda;

    lambdas = exp(linspace(log(maxlambda),-log(size(X,1)),ngridpt));
    BICs = zeros(1,ngridpt);
    LogLs = zeros(1,ngridpt);
    Dofs =zeros(1, ngridpt);
    tic;
    for j=1:ngridpt
        if j==1
            B0 = zeros(p,d-1);
        else
            B0 = B_hat;
        end
        [B_hat, stats] = mglm_sparsereg(X,Y,lambdas(j),'penalty',pen, ...
            'dist',dist,'B0',B0);
        BICs(j) = stats.BIC;
        LogLs(j) = stats.logL;
        Dofs(j) = stats.dof;
    end
    toc;
    
    % True signal versus estimated signal
    [bestbic,bestidx] = min(BICs);
    B_best = mglm_sparsereg(X,Y,lambdas(bestidx),'penalty',pen,'dist',dist);
    % display MSE of regularized estiamte
    display(norm(B_best-B,2)/sqrt(numel(B)));
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