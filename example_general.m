%% an example to test the accuracy
% min x1^2-4x1+2x2^2-12x2
% s.t.x1+x2=5
%     x1,x2>=0
clear all; clc;

P = [2,0;0,4];
q = [-4;-12];
A = [1,1];
b = 5;

[x, fval] = quadprog(P, q, [], [], A, b, zeros(2,1), []);
[x2, history] = quad_ADMM_general(P, q, 0, A, b, 1, 1);

%% an example to beat LP
clear all; clc;
