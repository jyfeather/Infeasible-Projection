% for box constrained problems
% linear programming is easy to solve
% quadratic programming & ADMM is diffcult to solve.

%% synthetic problem w/ box constraint
clear all; clc;
randn('state', 0);
rand('state', 0);

n = 1000;
% generate a well-conditioned positive definite matrix
% (for faster convergence)
P = rand(n, n);
P = P'*P;

q = randn(n,1);
r = 0;

l = randn(n,1);
u = randn(n,1);
lb = min(l,u);
ub = max(l,u);

% ADMM
fprintf('solve QP using ADMM\n');
tic
[x history] = quad_ADMM_bounded(P, q, r, lb, ub, 1.0, 1.0);
toc

% normal algirithm: interior point convex
fprintf('----------------------------\n');
fprintf('solve QP using quadprog\n');
tic
[x,fval] = quadprog(P, q, [], [], [], [], lb, ub);
toc

% linear programming
fprintf('----------------------------\n');
fprintf('solve LP using linprog\n');
f = randn(n,1);
tic
[x fval] = linprog(f,[],[],[],[],lb,ub);
toc