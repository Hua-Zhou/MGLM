%% Dirichlet-Multinomial regression and sparse regression
% A demo of Dirichlet-Multinomial regression and sparse regression

%% Generate Dirichlet-Multinomial random vectors from covariates

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
B = zeros(p,d);
nzidx = [1 3 5];
B(nzidx,:) = ones(length(nzidx),d);
alpha = exp(X*B);
batchsize = 25+unidrnd(25,n,1);
Y = dirmnrnd(batchsize,alpha);

zerorows = sum(Y,2);
Y=Y(zerorows~=0, :);
X=X(zerorows~=0, :);

%% Fit Dirichlet-Multinomial regression

tic;
[B_hat, stats_dm] = dirmnreg(X,Y);
toc;
display(B_hat);
display(stats_dm.se);
display(stats_dm);
% Wald test of predictor significance
display('Wald test p-values:');
display(stats_dm.wald_pvalue);

figure;
plot(stats_dm.logL_iter);
xlabel('iteration');
ylabel('log-likelihood');

%% Fit Dirichlet-Multinomial sparse regression - lasso/group/nuclear penalty

penalty = {'sweep','group','nuclear'};
ngridpt = 20;
dist = 'dirmn';

for i = 1:length(penalty)
    
    pen = penalty{i};
    [~, stats] = mglm_sparsereg(X,Y,inf,'penalty',pen,'dist',dist);
    maxlambda = stats.maxlambda;

    lambdas = exp(linspace(log(maxlambda),log(maxlambda/100),ngridpt));
    BICs = zeros(1,ngridpt);
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
    end
    toc;
    
    % True signal versus estimated signal
    [bestbic,bestidx] = min(BICs);
    lambdas(bestidx)
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