function gpe_imt_rk4(Comup_Domain,Grid_Size,g,Om,Nt)
global  g  mu  H0 X

mu0 = sqrt(g/2/pi)
Rtf = sqrt(2*mu0/sqrt(1-Om^2))
%%
N_grid    = prod(Grid_Size);
N_grid_xy = prod(Grid_Size(1:2));
grid_spacing = 2 * Comup_Domain ./ Grid_Size 
dx   = grid_spacing(1);
dy   = grid_spacing(2);
Nx   = Grid_Size(1);
Ny   = Grid_Size(2);   
dt  = 0.5*dx^2;
x    = -Comup_Domain(1) + [0:Nx-1]*dx + 0 * dx;
y    = -Comup_Domain(2) + [0:Ny-1]*dy + 0 * dy;
type = 1;

%% D2x
    unitAp1 = [];
     Grid_Size(3)=1
    for jj = 1 : Grid_Size(3)-1
        unitAp1 = [unitAp1 ones(1,N_grid_xy-Grid_Size(2)) zeros(1,Grid_Size(2))];
    end
    unitAp1 = [unitAp1 ones(1,N_grid_xy-1*Grid_Size(2))];
    Ap1 = sparse(1*Grid_Size(2)+1:N_grid,1:(N_grid-1*Grid_Size(2)),unitAp1,N_grid,N_grid);
    Ap0 = sparse(1:N_grid,1:N_grid,ones(1,N_grid),N_grid,N_grid);
   %
    unitAp1 = [];

    if type == 1
        for jj = 1 : Grid_Size(3)-1
            unitAp1 = [unitAp1 ones(1,Grid_Size(2)) zeros(1,N_grid_xy-Grid_Size(2))];
        end
        unitAp1 = [unitAp1 ones(1,Grid_Size(2))];
    else 
        unitAp1 = zeros(1,N_grid-1*(N_grid_xy-1*Grid_Size(2)));
    end
    Ap1p = sparse(1:N_grid-1*(N_grid_xy-1*Grid_Size(2)),N_grid_xy-1*Grid_Size(2)+1:N_grid,unitAp1,N_grid,N_grid);
    D2x =   1  * (Ap1 + Ap1' + Ap1p + Ap1p')...
           -2  * Ap0;
    D2x = D2x/dx^2;  
    Dx  =  -0.5  * (Ap1 - Ap1' + Ap1p - Ap1p') / dx;
    clear Ap1p unitAp1 Ap1
%% D2y
    unitAp1 = ones(1,N_grid-1);
    unitAp1(Grid_Size(2)-0:Grid_Size(2):end)=0;
    Ap1 = sparse(1+1:N_grid,1:N_grid-1,unitAp1,N_grid,N_grid);
        
    if type == 1
        unitAp1 = zeros(1,N_grid-Grid_Size(2)+1);
        unitAp1(1:Ny:end) = 1;
        Ap1p = sparse(1:N_grid-Grid_Size(2)+1,Grid_Size(2)-0:N_grid,unitAp1,N_grid,N_grid);
    end
    D2y =   1 * (Ap1 + Ap1' + Ap1p + Ap1p')...
           -2 * Ap0;

    D2y = D2y/dy^2;
    Dy = -0.5 * (Ap1 - Ap1' + Ap1p - Ap1p') / dy;
     clear Ap1p unitAp1 Ap1 Ap0
%%
[X,Y] = meshgrid(x,y);

trap = (0.5* (X.^2 + Y.^2));
Trap = sparse(1:N_grid,1:N_grid,trap(:),N_grid,N_grid);
XX = sparse(1:N_grid,1:N_grid,X(:),N_grid,N_grid);
YY = sparse(1:N_grid,1:N_grid,Y(:),N_grid,N_grid);
H0 =  -0.5 * (D2x + D2y) + Trap - 1i* Om * ( XX * Dy - YY * Dx);
H0 = sparse(H0);

%clear Trap
% Hamiltoian

%% Initial Conditions
aho     = 1;
%psi0    = (X+1i*Y).*exp(-(X.^2+Y.^2)/aho^2/2)/sqrt(pi*aho^2); % Initial Condition
data = load('wave_real_imag.txt');   % 假設是兩欄 [Re, Im]
psi0 = data(:,1) + 1i * data(:,2);   % 組成複數波函數
psi     = psi0(:);
Norm    = sum(abs(psi).^2)*dx*dy
psi = psi/sqrt(Norm);
nonlin  = sparse(1:N_grid,1:N_grid,g * (abs(psi(:)).^2)/Norm,N_grid,N_grid);
mu      = sum( (conj(psi(:)) .* ( (H0 *psi + g*abs(psi).^2.*psi)))) / sum(abs(psi(:)).^2)

%% Main Loop
parpool('local', 1);
u0 = psi;
clear psi
k         = 0; % Loop Parameters
tspan  = [0:1:Nt]*dt;
irt    = -1 % imaginary time propagation
%%i = 0 ;
tStart = cputime;
tic
while k <= Nt-1
    tspan_tmp = tspan(k+1:k+2);
    un = gpe_rk4(u0,tspan_tmp,dt,irt);
    un = un(end,:).';
    Norm = sum(abs(un).^2) * dx * dy;
    %mu = sum(conj(un) .* (H0 * un + g * abs(un).^2 / Norm .* un)) * dx * dy / Norm;
    %Etot = sum(conj(un) .* (H0*un + 0.5*g* abs(un).^2.*un))*dx*dy / Norm;
    %mu_t(k+1) = mu;
    u0 = un;
    cput = cputime - tStart;
    
    % each 100 steps save data
    if mod(k, 1) == 0
        tPause = cputime; %outputfiles cputime should be remove
        step_num = k; 
        
        % save data
        mu = sum(conj(un) .* (H0 * un + g * abs(un).^2 / Norm .* un)) * dx * dy / Norm;
        Etot = sum(conj(un) .* (H0*un + 0.5*g* abs(un).^2 / Norm .*un))* dx * dy / Norm;
        %u = reshape(un,Grid_Size(1:2));
        disp(['mu' num2str(k) ' = ' num2str(real(mu))]);
        disp(['Etot' num2str(k) ' = ' num2str(real(Etot))]);
        disp(['cput' num2str(k) ' = ' num2str(cput)]);
        %save_filename = ['gpe_benchmark-Lx=' num2str(Comup_Domain(1)*2) ...
        %                '-Nx=' num2str(Grid_Size(1)) ...
        %                '-Om=' num2str(Om) ...
        %                '-step=' num2str(step_num) '.mat'];
        %save(save_filename,'Norm', 'Etot', 'mu', 'cput');
        
        tStart = tStart + (cputime - tPause); % recover cput
    end
    k = k + 1;
end
toc
tEnd = cputime - tStart;
%%
u = reshape(un,Grid_Size(1:2));
Etot = sum(conj(un) .* (H0*un + 0.5*g* abs(un).^2 / Norm .*un))* dx * dy / Norm;
ur = real(un);
ui = imag(un);
mu_real = sum(ur.*((-0.5 * (D2x + D2y) + Trap)*ur + Om * ( XX * Dy - YY * Dx) * ui + g*abs(un).^2.*ur))/sum(ur.^2),
save(['gpe_benchmark-Lx=' num2str(Comup_Domain(1)*2) '-Nx=' num2str(Grid_Size(1)) '-Om=' num2str(Om) '.mat'],'u','x','y','Norm','mu','Etot','mu_real', 'cput')
%%


 %% RK-4
function un=rk4(h,u,time,irt)
    u=[u(:)];
    k1=hamil(u,time,irt);
    k2=hamil(u+0.5*k1*h,time+0.5*h,irt); 
    k3=hamil(u+0.5*k2*h,time+0.5*h,irt);
    k4=hamil(u+k3*h,time+h,irt);
    un=u+h*(k1+2*k2+2*k3+k4)/6;

%% SPGPE Sovler
function uall = gpe_rk4(u0,TimeSpan,h,irt)
global  g  mu  H0 X
    uall=zeros(length(TimeSpan),length(u0(:)));
    uall(1,:)=u0(:);
    savenumber=1; tc = TimeSpan(1); SaveStep = TimeSpan(2)-TimeSpan(1);
    while savenumber<length(TimeSpan)
        tspanc=TimeSpan(savenumber):h:TimeSpan(savenumber)+SaveStep;
        for j=1:length(tspanc)
            t=tspanc(j);
            un=rk4(h,u0,t,irt);
            %u1n=reshape(un(:),size(X));    
            u0= un(:);%[u1n(:).' ];
            if any(isnan(u0))
                error('Divergence. The time step is not small enough.');
                break
            end
        end
        t;
        uall(savenumber+1,:)=u0(:);
        savenumber=savenumber+1;
    end

%% Hamiltonian
function dudt=hamil(u,time,irt)
global  g  mu  H0
u = u (:);
Lu1=  H0*u+ (g * abs(u).^2 -mu).*u;

du1dt= irt* Lu1; %du1dth = fft(du1dt); du1dth(Etall-Ecut>0)=0;

dudt=[du1dt(:) ];
