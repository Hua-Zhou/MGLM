%% Generalized Dirichlet-Multinomial regression and sparse regression
% A demo of gen. Dirichlet-Multinomial regression and sparse regression

%% Generate generalized Dirichlet-Multinomial random vectors from covariates

clear;
% reset random seed
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
% sample size
n = 500;
% # covariates
p = 15;
% # bins
d = 5;
% design matrix
X = randn(n,p); 
% true regression coefficients
A = zeros(p,d-1);
B = zeros(p,d-1);
nzidx = [1 3 5];
A(nzidx,:) = 0.5.*ones(length(nzidx),d-1);
B(nzidx,:) = 0.5.*ones(length(nzidx),d-1);
alpha = exp(X*A);
beta = exp(X*B);
batchsize = 25+unidrnd(25,n,1);
Y = gendirmnrnd(batchsize,alpha, beta);

%% Fit generalized Dirichlet-Multinomial regression

tic;
[Bhat1,Bhat2,stats_gdm] = gendirmnreg(X,Y);
toc;
display(Bhat1);
display(Bhat2);
display(stats_gdm);
display(stats_gdm.se);
display(stats_gdm.wald_pvalue);

%% Fit generalized Dirichlet-Multinomial sparse regression - lasso/group/nuclear penalty
penalty = {'sweep','group','nuclear'};
ngridpt = 10;
dist = 'gendirmn';

for i = 1:length(penalty)
    
    pen = penalty{i};
    [~, stats] = mglm_sparsereg(X,Y,inf,'penalty',pen,'dist',dist);
    maxlambda = stats.maxlambda;

    lambdas = exp(linspace(log(maxlambda),log(maxlambda/100),ngridpt));
    BICs = zeros(1,ngridpt);
    tic;
    for j=1:ngridpt
        if j==1
            B0 = zeros(p,2*(d-1));
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
    imshow(mat2gray(-[A,B])); title('True B');
    subplot(1,3,3);
    imshow(mat2gray(-B_best)); title([pen ' estimate']);

end
