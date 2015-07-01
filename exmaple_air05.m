%%
% http://miplib.zib.de/miplib2003/miplib2003.php
% air 05
clear all; clc;

origin = readmps('data\air05.mps');

n = 7195;
p = 426;

tmp = origin.A;
C = tmp(1,:);
A = tmp(2:end,:);
b = origin.rhs(2:end);

%% linear programming
tic;
[x, fval] = linprog(C,[],[],A,b,zeros(n,1),[]);
toc;

%% quadratic programming w/ ADMM
guess_solution = randn(n,1);
P = eye(n);
q = -guess_solution;

tic;
[x, history] = quad_ADMM_general(P, q, 0, A, b, 1, 1);
toc;