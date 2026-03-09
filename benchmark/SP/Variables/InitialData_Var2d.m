%% Computation of the initial data
%% INPUTS:
%%          Method: Structure containing variables concerning the method (see Method_Var2d.m)
%%          Geometry2D: Structure containing variables concerning the geometry of the problem in 2D (see Geometry2D_Var2d.m)
%%          Physics2D: Structure containing variables concerning the physics of the problem in 2D (see Physics2D_Var2d.m)
%% INPUTS(OPTIONAL):
%%          InitialData_Choice: 1=Gaussian, 2=Thomas-Fermi, 3=CNSP-CNFG, 4=User-defined
%%          X0,Y0: Gaussian/TF center
%%          gamma_x,gamma_y: Gaussian parameters
%% OUTPUT:
%%          Phi_0: Initial wavefunction (cell array)

function [Phi_0] = InitialData_Var2d(varargin)

%% === 1. Analysis of the inputs ===
Analyse_Var = inputParser;
Analyse_Var.addRequired('Method');
Analyse_Var.addRequired('Geometry2D');
Analyse_Var.addRequired('Physics2D');
Analyse_Var.addOptional('InitialData_Choice', 1);
Analyse_Var.addOptional('X0', 0);
Analyse_Var.addOptional('Y0', 0);
Analyse_Var.addOptional('gamma_x', 1);
Analyse_Var.addOptional('gamma_y', 1);

Analyse_Var.parse(varargin{:});
Method = Analyse_Var.Results.Method;
Geometry2D = Analyse_Var.Results.Geometry2D;
Physics2D = Analyse_Var.Results.Physics2D;
InitialData_Choice = Analyse_Var.Results.InitialData_Choice;
X0 = Analyse_Var.Results.X0;
Y0 = Analyse_Var.Results.Y0;
gamma_x = Analyse_Var.Results.gamma_x;
gamma_y = Analyse_Var.Results.gamma_y;

if isscalar(X0)
    X0 = X0 * ones(1, Method.Ncomponents);
end
if isscalar(Y0)
    Y0 = Y0 * ones(1, Method.Ncomponents);
end

%% === 2. Initialization ===
Phi_0 = cell(1, Method.Ncomponents);

%% === 3. Initial data computation ===
if (InitialData_Choice == 1)
    %% --- Centered Gaussian ---
    for n = 1:Method.Ncomponents
        Phi_0{n} = GaussianInitialData2d(Geometry2D, Physics2D, gamma_x, gamma_y, X0(n), Y0(n));
        Phi_0{n} = Phi_0{n} / L2_norm2d(Phi_0{n}, Geometry2D);
    end

elseif (InitialData_Choice == 2) && (Physics2D.Beta ~= 0)
    %% --- Thomas-Fermi approximation ---
    for n = 1:Method.Ncomponents
        Phi_0{n} = Thomas_Fermi2d(gamma_x, gamma_y, Physics2D.Beta, ...
            Physics2D.Potential_function{n,n}(Geometry2D.X-X0(n), Geometry2D.Y-Y0(n)) + ...
            Physics2D.TimePotential_function{n,n}(0, Geometry2D.X-X0(n), Geometry2D.Y-Y0(n)));
        Phi_0{n} = Phi_0{n} / L2_norm2d(Phi_0{n}, Geometry2D);
    end

elseif (InitialData_Choice == 2) && (Physics2D.Beta == 0)
    %% --- TF invalid, fall back to Gaussian ---
    fprintf('Cannot compute TF with Beta=0. Using Gaussian instead.\n');
    for n = 1:Method.Ncomponents
        Phi_0{n} = GaussianInitialData2d(Geometry2D, Physics2D, gamma_x, gamma_y, X0(n), Y0(n));
        Phi_0{n} = Phi_0{n} / L2_norm2d(Phi_0{n}, Geometry2D);
    end

elseif (InitialData_Choice == 3)
    %% --- CNSP / CNGF scheme ---
    Print.Print = 1;
    Print.Type = 2;
    Print.Evo = 5;
    Print.Draw = 1;
    Method.Computation = 'Ground';
    Method.Type = 'BESP';
    Method.Deltat = 1e-1;
    Method.Precond = 'ThomasFermi';
    Outputs = OutputsINI_Var2d(Method);
    Figure = Figure_Var2d;

    if (Physics2D.Beta ~= 0)
        fprintf('Computing initial data with BESP (Thomas-Fermi start)\n');
        for n = 1:Method.Ncomponents
            Phi_0{n} = Thomas_Fermi2d(gamma_x, gamma_y, Physics2D.Beta, ...
                Physics2D.Potential_function{n,n}(Geometry2D.X, Geometry2D.Y));
        end
        Phi_0 = BESP_CNGF2d(Phi_0, Method, Geometry2D, Physics2D, Outputs, Print, Figure);
    else
        fprintf('Computing initial data with BESP (Gaussian start)\n');
        for n = 1:Method.Ncomponents
            Phi_0{n} = GaussianInitialData2d(Geometry2D, Physics2D, gamma_x, gamma_y, X0(n), Y0(n));
        end
        Phi_0 = BESP_CNGF2d(Phi_0, Method, Geometry2D, Physics2D, Outputs, Print, Figure);
    end

    for n = 1:Method.Ncomponents
        Phi_0{n} = Phi_0{n} / L2_norm2d(Phi_0{n}, Geometry2D);
    end

elseif (InitialData_Choice == 4)
    %% --- User-defined initial wavefunction from text file ---
    fprintf('Reading user-defined initial wave function from text file(s)...\n');

    if ~isfield(Physics2D, 'UserDefinedFile')
        error('Physics2D.UserDefinedFile must be defined when InitialData_Choice == 4.');
    end

    for n = 1:Method.Ncomponents
        filename = Physics2D.UserDefinedFile{n};
        fprintf('  Loading %s ...\n', filename);

        data = load(filename); % Expect: [Re, Im]
        if size(data, 2) < 2
            error('File %s must have two columns: real and imaginary parts.', filename);
        end

        psi = data(:,1) + 1i * data(:,2);

        [Ny, Nx] = size(Geometry2D.X);
        if numel(psi) ~= Nx * Ny
            error('File %s size (%d) does not match grid size (%d×%d).', ...
                  filename, numel(psi), Nx, Ny);
        end

        % numpy.meshgrid uses row-major (y,x) flattening
        Phi_0{n} = reshape(psi, Nx, Ny);  
        Phi_0{n} = Phi_0{n} ./ L2_norm2d(Phi_0{n}, Geometry2D);
        %L2_norm2d(Phi_0{n}, Geometry2D)
        %pause
    end

else
    error('Unknown InitialData_Choice = %d', InitialData_Choice);
end

end

