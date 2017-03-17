%% Negative multinomial distribution
% A demo of random number generation, density evaluation, and distribution
% fitting for the negative mutlinomial distribution

%% Generate negative multinomial random vectors

clear;
% reset random seed
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
% sample size
n = 100;
% # bins
d = 5;
% prob parameter
prob = repmat(1/(d+1),1,d+1);
% overdispersion
b = 2;
% generate random vectors
X = negmnrnd(prob,b,n);

%% Evaluate negative multinomial pdf at true parameter value

logL = negmnpdfln(X,prob,b);
display(sum(logL));

%% Fit negative multinomial distribution

tic;
[p_hat, b_hat, stats] = negmnfit(X);
toc;
display([p_hat, b_hat]);
display(stats.se');
display(stats);

figure;
plot(stats.logL_iter);
xlabel('iteration');
ylabel('log-likelihood');