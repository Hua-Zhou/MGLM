%% Generalized Dirichlet-Multinomial distribution
% A demo of random number generation, density evaluation, and distribution
% fitting for the generalized Dirichlet-Mutlinomial distribution

%% Generate generalized Dirichlet-Multinomial random vectors

clear;
% reset random seed
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
% sample size
n = 100;
% # bins
d = 5;
% true parameter values
alpha = 1:d-1;
beta = d-1:-1:1;
% set batch sizes (ranges from 25 to 50)
batchsize = 25+unidrnd(25,n,1);
% generate random vectors
X = gendirmnrnd(batchsize,alpha,beta,n);

%% Evaluate gen. Dirichlet-Multinomial (log) pdf at true parameter value

logL = gendirmnpdfln(X,alpha,beta);
display(sum(logL));

%% Fit generalized Dirichlet-Multinomial distribution

tic;
[alpha_hat, beta_hat, stats_gdm] = gendirmnfit(X);
toc;
display([alpha_hat, beta_hat]);
display(stats_gdm.se);
display(stats_gdm);
