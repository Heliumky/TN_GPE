%%% This file is an example of how to use GPELab (FFT version)

%% GROUND STATE COMPUTATION WITH A ROTATING TERM

%-----------------------------------------------------------
% Setting the data
%-----------------------------------------------------------

%% Setting the method and geometry
Computation = 'Ground';
Ncomponents = 1;
Type = 'CNFD';
%Type = 'BESP';
Deltat = 1e-2;
Stop_time = [];
Stop_crit = {'MaxNorm',1e-4};
Iterative_maxit = 15000;

Method = Method_Var2d(Computation, Ncomponents, Type, Deltat, Stop_time, Stop_crit, Iterative_maxit);
Method.Solver_FD = 1;  % 使用直接求解器而不是迭代法
%Method.Solver_BESP = 1;
%Method.Iterative_maxit = 1000;
xmin = -21;
xmax = 21;
ymin = -21;
ymax = 21;
Nx = 2^11;
Ny = 2^11;
Geometry2D = Geometry2D_Var2d(xmin,xmax,ymin,ymax,Nx,Ny);

%% Setting the physical problem
Delta = 0.5;
Beta = 100;
Omega = 0.946;
Physics2D = Physics2D_Var2d(Method, Delta, Beta, Omega);
Physics2D = Dispersion_Var2d(Method, Physics2D);
Physics2D = Potential_Var2d(Method, Physics2D);
Physics2D = Nonlinearity_Var2d(Method, Physics2D);
Physics2D = Gradientx_Var2d(Method, Physics2D);
Physics2D = Gradienty_Var2d(Method, Physics2D);
%% Setting the initial data
Physics2D.UserDefinedFile = {'wave_real_imag.txt'};
InitialData_Choice = 4;
Phi_0 = InitialData_Var2d(Method, Geometry2D, Physics2D, InitialData_Choice);

%% Setting informations and outputs
Evo_outputs = 100;       % 每100步计算输出
save_solution = 0;       % 保存波函数
Outputs = OutputsINI_Var2d(Method, Evo_outputs, save_solution);

% 禁用打印系统
Printing = 0;
Evo = 100;
Draw = 0;
Print = Print_Var2d(Printing, Evo, Draw);
%-----------------------------------------------------------
% Launching simulation
%-----------------------------------------------------------

[Phi, Outputs] = GPELab2d(Phi_0, Method, Geometry2D, Physics2D, Outputs, [], Print);
save('GPELab_results.mat', 'Outputs');
fprintf('数据已保存到 GPELab_CNFD_results.mat\n');
