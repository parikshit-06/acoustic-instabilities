% Simulation Parameters
N = 1000;      % number of timesteps
T = 40;        % total time of experiment
dt = (T/N);    % time step
xf = 0.3;      % feedback location

% System Parameters
g = 1.4;
c0 = 399.6;    % speed of sound
u0 = 0.5;      % mean velocity of air
M = u0/c0;     % Mach number
J = 3;         % number of Galerkin modes

% Parameter Ranges
ks = 0.2:0.01:1.5;     % non-dimensional heater power
tau = 0.2:0.01:0.8;    % time-lag

% Preallocate Arrays
n1 = length(ks);
n3 = length(tau);
u_rms1 = zeros(n3, n1);
u_rms2 = zeros(n3, n1);
y1 = zeros(n3, J, N);
y2 = zeros(n3, J, N);

% Forward Simulation
for i = 1:n3
    y1(i, 1, 1) = 0.18;  % Initial condition
    u1 = zeros(n1, N);
    
    for k = 1:n1
        % Initial velocity calculation
        for j = 1:J
            u1(k, 1) = u1(k, 1) + y1(i, j, 1) * cos(j * pi * xf);
        end
        
        n2 = round(tau(i) / dt);
        
        % Fourth-order Runge-Kutta method for t < tau
        for n = 1:n2
            for j = 1:J
                [k1, l1] = equations1(y1(i, j, n), y2(i, j, n), j);
                [k2, l2] = equations1(y1(i, j, n) + 0.5 * dt * k1, y2(i, j, n) + 0.5 * dt * l1, j);
                [k3, l3] = equations1(y1(i, j, n) + 0.5 * dt * k2, y2(i, j, n) + 0.5 * dt * l2, j);
                [k4, l4] = equations1(y1(i, j, n) + dt * k3, y2(i, j, n) + dt * l3, j);
                
                y1(i, j, n + 1) = y1(i, j, n) + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
                y2(i, j, n + 1) = y2(i, j, n) + (dt / 6) * (l1 + 2 * l2 + 2 * l3 + l4);
                
                u1(k, n + 1) = u1(k, n + 1) + y1(i, j, n + 1) * cos(j * pi * xf);
            end
        end
        
        % Fourth-order Runge-Kutta method for t > tau
        for n = n2 + 1:N - 1
            for j = 1:J
                [k1, l1] = equations2(y1(i, j, n), y2(i, j, n), j, ks(k), u1(k, n - n2), xf);
                [k2, l2] = equations2(y1(i, j, n) + 0.5 * dt * k1, y2(i, j, n) + 0.5 * dt * l1, j, ks(k), u1(k, n - n2), xf);
                [k3, l3] = equations2(y1(i, j, n) + 0.5 * dt * k2, y2(i, j, n) + 0.5 * dt * l2, j, ks(k), u1(k, n - n2), xf);
                [k4, l4] = equations2(y1(i, j, n) + dt * k3, y2(i, j, n) + dt * l3, j, ks(k), u1(k, n - n2), xf);
                
                y1(i, j, n + 1) = y1(i, j, n) + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
                y2(i, j, n + 1) = y2(i, j, n) + (dt / 6) * (l1 + 2 * l2 + 2 * l3 + l4);
                
                u1(k, n + 1) = u1(k, n + 1) + y1(i, j, n + 1) * cos(j * pi * xf);
            end
        end

        % Update initial conditions and calculate RMS
        y1(i, :, 1) = y1(i, :, N);
        y2(i, :, 1) = y2(i, :, N);
        u_rms1(i, k) = rms(u1(k, :));
    end
end

% Backward Simulation
for i = 1:n3
    u2 = zeros(n1, N);
    
    for k = 1:n1
        % Initial velocity calculation (fixed typo with J instead of j)
        for j = 1:J
            u2(n1 - k + 1, 1) = u2(n1 - k + 1, 1) + y1(i, j, 1) * cos(j * pi * xf);
        end
        
        n2 = round(tau(i) / dt);
        
        % Fourth-order Runge-Kutta method for t < tau
        for n = 1:n2
            for j = 1:J
                [k1, l1] = equations1(y1(i, j, n), y2(i, j, n), j);
                [k2, l2] = equations1(y1(i, j, n) + 0.5 * dt * k1, y2(i, j, n) + 0.5 * dt * l1, j);
                [k3, l3] = equations1(y1(i, j, n) + 0.5 * dt * k2, y2(i, j, n) + 0.5 * dt * l2, j);
                [k4, l4] = equations1(y1(i, j, n) + dt * k3, y2(i, j, n) + dt * l3, j);
                
                y1(i, j, n + 1) = y1(i, j, n) + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
                y2(i, j, n + 1) = y2(i, j, n) + (dt / 6) * (l1 + 2 * l2 + 2 * l3 + l4);
                
                u2(n1 - k + 1, n + 1) = u2(n1 - k + 1, n + 1) + y1(i, j, n + 1) * cos(j * pi * xf);
            end
        end
        
        % Fourth-order Runge-Kutta method for t > tau
        for n = n2 + 1:N - 1
            for j = 1:J
                [k1, l1] = equations2(y1(i, j, n), y2(i, j, n), j, ks(n1 - k + 1), u2(n1 - k + 1, n - n2), xf);
                [k2, l2] = equations2(y1(i, j, n) + 0.5 * dt * k1, y2(i, j, n) + 0.5 * dt * l1, j, ks(n1 - k + 1), u2(n1 - k + 1, n - n2), xf);
                [k3, l3] = equations2(y1(i, j, n) + 0.5 * dt * k2, y2(i, j, n) + 0.5 * dt * l2, j, ks(n1 - k + 1), u2(n1 - k + 1, n - n2), xf);
                [k4, l4] = equations2(y1(i, j, n) + dt * k3, y2(i, j, n) + dt * l3, j, ks(n1 - k + 1), u2(n1 - k + 1, n - n2), xf);
                
                y1(i, j, n + 1) = y1(i, j, n) + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
                y2(i, j, n + 1) = y2(i, j, n) + (dt / 6) * (l1 + 2 * l2 + 2 * l3 + l4);
                
                u2(n1 - k + 1, n + 1) = u2(n1 - k + 1, n + 1) + y1(i, j, n + 1) * cos(j * pi * xf);
            end
        end

        % Update initial conditions and calculate RMS
        y1(i, :, 1) = y1(i, :, N);
        y2(i, :, 1) = y2(i, :, N);
        u_rms2(i, n1 - k + 1) = rms(u2(n1 - k + 1, :));
    end
end

% Plotting
% Bifurcation plot for tau = 0.2
for i = 1:n3
    if abs(tau(i) - 0.2) < 1e-6  % Avoid floating-point precision issues
        figure(1);
        scatter(ks, u_rms1(i, :), 'MarkerEdgeColor', 'k', 'LineWidth', 2); % Purple
        hold on;
        scatter(ks, u_rms2(i, :), 'MarkerEdgeColor', 'g', 'LineWidth', 1); % Orange
        xlabel('K');
        ylabel('|u_1|');
        legend('Forward', 'Backward')
        title('Bifurcation Plot')
        annotation('textarrow', [0.7 0.6], [0.3 0.15], 'String', 'Hopf Bifurcation')
        annotation('textarrow', [0.25 0.3], [0.65 0.55], 'String', 'Fold Bifurcation')
        grid on;
    end
end

% 3D Bifurcation plot
figure(2);
waterfall(ks, tau, u_rms1);
hold on;
waterfall(ks, tau, u_rms2);
colormap turbo  % Change color scheme to 'turbo'
xlabel('K')
ylabel('tau');
zlabel('|u_1|')
legend('Forward', 'Backward')
title('3D Bifurcation Plot')
annotation('textarrow', [0.6 0.45], [0.3 0.15], 'String', 'Hopf Bifurcation')
annotation('textarrow', [0.2 0.25], [0.6 0.5], 'String', 'Fold Bifurcation')
colorbar

% Nested Functions
function [f1, f2] = equations1(y1, y2, j)
    kj = j * pi;
    wj = kj;
    w1 = pi;
    c1 = 0.1; 
    c2 = 0.06;
    
    zetaj = (1 / (2 * pi)) * (c1 * (wj / w1) + c2 * sqrt(w1 / wj));
    
    f1 = y2;
    f2 = -2 * zetaj * wj * y2 - (kj^2) * y1;
end

function [f1, f2] = equations2(y1, y2, j, K, u, xf)
    kj = j * pi;
    wj = kj;
    w1 = pi;
    c1 = 0.1; 
    c2 = 0.06;
    
    zetaj = (1 / (2 * pi)) * (c1 * (wj / w1) + c2 * sqrt(w1 / wj));
    
    f1 = y2;
    f2 = -2 * zetaj * wj * y2 - (kj^2) * y1 - j * pi * K * (abs(sqrt((1/3) + u)) - sqrt(1/3)) * sin(j * pi * xf);
end