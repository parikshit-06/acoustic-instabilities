%% Simulation Parameters
N = 1000;      % number of timesteps
T = 80;         % for fig 5a, 5b, 5c, 5d, 6b, 6c, 6d
dt = (T/N);     % time step

%% System Parameters
g = 1.4;        % specific heat ratio
c0 = 399.6;     % speed of sound
u0 = 0.5;       % mean velocity of air
M = u0/c0;      % Mach number

%% Configurable Parameters
%% K values
%K = 0.05;       % fig4a
%K = 0.3;        % fig4b, 4d
%K = 0.25;       % fig5a, 5b, 5c, 5d
K = 0.01;       % fig6a, 6b, 6c, 6d

xf = 0.29;      % flame position 

%% Tau values
%tau = 0.2;      % fig4a
%tau = 0.5;      % fig4b, 4d, 
%tau = 0.75;     % fig5a, 5b, 5c, 5d
tau = 0.45*pi;  % fig6a, 6b, 6c, 6d

%% Initial Conditions for y1 (only one of three)
%y1_init = 0.15;  % fig 4a
%y1_init = 0.2;   % fig4b, 4d
y1_init = 0.18;   % fig5a, 5b, 5c, 5d, fig6a, 6b, 6c, 6d

% Calculate delay index
n1 = round(tau/dt);

%% Initialize Arrays
J = 3;             % number of Galerkin modes
u = zeros(1, N);
p = zeros(1, N);
y1 = zeros(J, N);
y2 = zeros(J, N);

%% Set Initial Conditions
y1(:,1) = 0;
y1(1,1) = y1_init; % Use selected initial condition
y2(:,1) = 0;
u(1) = y1(1,1)*cos(pi*xf);
p(1) = 0;

%% Precompute constants for equations
kj = (1:J)'*pi;
wj = kj;
w1 = pi;
c1 = 0.1;
c2 = 0.06;
zetaj = (1/(2*pi))*(c1*(wj/w1) + c2*sqrt(w1./wj));

%% Precompute mode shapes
cos_j_pi_xf = cos((1:J)'*pi*xf);
sin_j_pi_xf = sin((1:J)'*pi*xf);

%% Time Integration
% First part: before delay (t < tau)
for n = 1:n1
    for j = 1:J
        % RK4 integration steps
        [k1, l1] = equations1(y1(j,n), y2(j,n), j, zetaj(j));
        [k2, l2] = equations1(y1(j,n) + 0.5*dt*k1, y2(j,n) + 0.5*dt*l1, j, zetaj(j));
        [k3, l3] = equations1(y1(j,n) + 0.5*dt*k2, y2(j,n) + 0.5*dt*l2, j, zetaj(j));
        [k4, l4] = equations1(y1(j,n) + dt*k3, y2(j,n) + dt*l3, j, zetaj(j));
        
        y1(j,n+1) = y1(j,n) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
        y2(j,n+1) = y2(j,n) + (dt/6) * (l1 + 2*l2 + 2*l3 + l4);
    end
    
    % Calculate u and p for this timestep
    u(n+1) = sum(y1(:,n+1).*cos_j_pi_xf);
    p(n+1) = sum(y2(:,n+1).*(-g*M./kj).*sin_j_pi_xf);
end

% Second part: after delay (t > tau)
for n = n1+1:N-1
    for j = 1:J
        % RK4 integration steps
        [k1, l1] = equations2(y1(j,n), y2(j,n), j, K, u(n-n1), xf, zetaj(j));
        [k2, l2] = equations2(y1(j,n) + 0.5*dt*k1, y2(j,n) + 0.5*dt*l1, j, K, u(n-n1), xf, zetaj(j));
        [k3, l3] = equations2(y1(j,n) + 0.5*dt*k2, y2(j,n) + 0.5*dt*l2, j, K, u(n-n1), xf, zetaj(j));
        [k4, l4] = equations2(y1(j,n) + dt*k3, y2(j,n) + dt*l3, j, K, u(n-n1), xf, zetaj(j));
        
        y1(j,n+1) = y1(j,n) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
        y2(j,n+1) = y2(j,n) + (dt/6) * (l1 + 2*l2 + 2*l3 + l4);
    end
    
    % Calculate u and p for this timestep
    u(n+1) = sum(y1(:,n+1).*cos_j_pi_xf);
    p(n+1) = sum(y2(:,n+1).*(-g*M./kj).*sin_j_pi_xf);
end

%% Energy Calculation
e = ((0.5*(p.^2) + 0.5*(g*M*u).^2)/((g*M)^2));
ee = envelope(e, 200, "peak");

%% Plotting
t = linspace(0,T,N);

% Main velocity plot
figure(1);
plot(t, u, 'b', 'LineWidth', 2); 
title('Variation of u''', 'FontWeight', 'bold');
xlabel('t (s)');
ylabel('u'' ');
fontsize(gcf, scale=1.5);

% Galerkin mode plots
figure(2);
plot(t, y1(1,:), 'r', 'LineWidth', 2); 
xlabel('t (s)');
ylabel('\eta_1');
title('Variation of \eta_1 (t)', 'FontWeight', 'bold');
fontsize(gcf, scale=1.5);

figure(3);
plot(t, y1(2,:), 'g', 'LineWidth', 2); 
xlabel('t (s)');
ylabel('Variation of \eta_2');
title('\eta_2 (t)', 'FontWeight', 'bold');
fontsize(gcf, scale=1.5);

figure(4);
plot(t, y1(3,:), 'm', 'LineWidth', 2); 
xlabel('t (s)');
ylabel('\eta_3');
title('Variation of \eta_3 (t)', 'FontWeight', 'bold');
fontsize(gcf, scale=1.5);

% Energy plot
figure(5);
plot(t, ee, 'k', 'LineWidth', 2); 
title('Energy Fluctuations Amplitude', 'FontWeight', 'bold');
xlabel('t (s)');
ylabel('Energy/(\gamma M)^2 ');
legend('Non Linear', 'Location', 'best');
fontsize(gcf, scale=1.5);

% Combined plot
figure(6)
tiledlayout(2,2)

nexttile
plot(t, u, 'b', 'LineWidth', 2); 
title('Variation of u''', 'FontWeight', 'bold');
xlabel('t (s)');
ylabel('u'' ');

nexttile
plot(t, y1(1,:), 'r', 'LineWidth', 2); 
xlabel('t (s)');
ylabel('\eta_1');
title('Variation of \eta_1 (t)', 'FontWeight', 'bold');

nexttile
plot(t, y1(2,:), 'g', 'LineWidth', 2); 
xlabel('t (s)');
ylabel('\eta_2');
title('Variation of \eta_2 (t)', 'FontWeight', 'bold');

nexttile
plot(t, y1(3,:), 'm', 'LineWidth', 2); 
xlabel('t (s)');
ylabel('\eta_3');
ylim([-0.4 0.4]);
title('Variation of \eta_3 (t)', 'FontWeight', 'bold');


%% Functions
function [f1, f2] = equations1(y1, y2, j, zetaj)
    kj = j*pi;
    wj = kj;
    
    f1 = y2;
    f2 = -(kj^2) * y1 - 2*zetaj*wj*y2;
end

function [f1, f2] = equations2(y1, y2, j, K, u, xf, zetaj)
    kj = j*pi;
    wj = kj;
    
    f1 = y2;
    f2 = -(kj^2) * y1 - 2*zetaj*wj*y2 - ((2*j*pi*K))*(sqrt(abs((1/3) + u))/sqrt(1/3))*sin(j*pi*xf);
end