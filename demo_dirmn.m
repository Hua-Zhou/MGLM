%% Dirichlet-Multinomial distribution
% A demo of random number generation, density evaluation, and distribution
% fitting for the Dirichlet-Mutlinomial distribution

%% Generate Dirichlet-Multinomial random vectors

clear;
% reset random seed
s = RandStream('mt19937ar','Seed',2);
RandStream.setGlobalStream(s);
% sample size
n = 100;
% # bins
d = 5;
% true parameter values
alpha = 1:d;
% set batch sizes (ranges from 25 to 50)
batchsize = 25+unidrnd(25,n,1);
% generate random vectors
X = dirmnrnd(batchsize,alpha,n);

%% Evaluate Dirichlet-Multinomial (log) pdf at true parameter value

logL = dirmnpdfln(X,alpha);
display(sum(logL));

%% Fit Dirichlet-Multinomial distribution

tic;
[alpha_hat, stats_dm] = dirmnfit(X);
toc;
display([alpha_hat, stats_dm.se]);
display(stats_dm);

figure;
plot(stats_dm.logL_iter);
xlabel('iteration');
ylabel('log-likelihood');
