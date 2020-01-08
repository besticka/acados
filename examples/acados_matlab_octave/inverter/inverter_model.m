function model = inverter_model()

import casadi.*


%% system dimensions
nx = 6;
nu = 2;

%% system parameters
f_B        = 150e3;
T_B        = 1/f_B;
f_INV        = 150e3;
T_INV        = 1/f_INV;
wG       = 2*pi*50;

v_in     = 50;

x0 = zeros(nx, 1);
%x0(1) = 
x0(2) = 0;
%x0(3) = 
%x0(4) = 
%x0(5) = 
x0(6) = 50;

%
L_B       = 80e-6;
C_DC      = 680e-6;
L_M       = 150e-6;
C_F       = 16e-6;
L_L       = 90e-6;


% cost
ny = 7;
ny_e = 5;
V_x = zeros(ny,nx);
V_x(1,1) = 1;
V_x(2,2) = 1;
V_x(3,3) = 1;
V_x(4,4) = 1;
V_x(5,5) = 1;
% V_x(6,6) = 1;
V_x_e = zeros(ny_e,nx);
V_x_e(1,1) = 1;
V_x_e(2,2) = 1;
V_x_e(3,3) = 1;
V_x_e(4,4) = 1;
V_x_e(5,5) = 1;
V_u = zeros(ny,nu);
V_u(6,1) = 1;
V_u(7,2) = 1;
Q = 5*diag([.0; 10; 1; 1; 1]);
R = 1e-5*diag([1; 1]); 
W = zeros(ny, ny);
W(1:5,1:5) = Q;
W(6:7,6:7) = R;
%W_e = Q*.1;% Q*0;
W_e = T_INV*Q*10000; % scaled with shooting node length, to have the same contribution in the  QP solver as W
x_ref = zeros(5,1);

x_ref(1) = 50;
x_ref(2) = 200;
x_ref(3) = -50;
x_ref(4) = 50;
x_ref(5) = -50;
y_ref = [x_ref; .5*ones(nu,1)];
y_ref_e = x_ref;

% constraints
nbu = 2;
Jbu = eye(nu,nu);
lbu = [0; 0];
ubu = [1; 1];


%% named symbolic variables
iB      = SX.sym('iB',1,1);
vDC     = SX.sym('vDC',1,1);
iM      = SX.sym('iM',1,1);
vC      = SX.sym('vC',1,1);
iL      = SX.sym('iL',1,1);
vG      = SX.sym('vG',1,1);
sym_x       = [iB; vDC; iM; vC; iL; vG];

dB      = SX.sym('dB',1,1);
dINV    = SX.sym('dINV',1,1);
sym_u       = [dB; dINV];

sym_xdot = SX.sym('xdot', nx, 1);

expr_f_expl = [(v_in-dB*vDC)/L_B;      %iB
		(dB*iB-dINV*iM)/C_DC;   %vDC
		(dINV*vDC - vC)/L_M;    %iM
		(iM-iL)/C_F;            %vC
		(vC-vG)/L_L;            %iL
		0];                     %vG
expr_f_impl = expr_f_expl - sym_xdot;



x_ss = [
    50.693
   200.806
    50.292
    49.988
    50.294
    50.000
];

u_ss = [
   0.25364
   0.24882
];

A_ss = [
   0.99976  -0.06340   0.00023  -0.00001   0.00000  -0.00000
   0.00746   0.99964  -0.00712   0.00047  -0.00020   0.00001
   0.00012   0.03227   0.91959  -0.12367   0.08029  -0.00604
   0.00005   0.01998   1.15939   0.78589  -1.15944   0.13382
   0.00000   0.00150   0.13382   0.20612   0.86618  -0.21216
   0.00000   0.00000   0.00000   0.00000   0.00000   1.00000
];

B_ss = [
  -50.24601    0.04895
    1.30013   -1.57559
    0.02230   26.02211
    0.00943   16.11296
    0.00054    1.21182
    0.00000    0.00000
];

Q_ss = 0.1*eye(6);
Q_ss(1:5,1:5) = W(1:5,1:5);
R_ss = W(6:7,6:7);

Q_ss = T_INV*Q_ss;
R_ss = T_INV*R_ss;

P = eye(6);
for ii=1:1000
	P = Q_ss + A_ss'*P*A_ss - A_ss'*P*B_ss*inv(R_ss+B_ss'*P*B_ss)*B_ss'*P*A_ss;
end
%W_e = P(1:5,1:5);



%% populate structure
model.nx = nx;
model.nu = nu;
model.ny = ny;
model.ny_e = ny_e;
model.nbu = nbu;
model.sym_x = sym_x;
model.sym_xdot = sym_xdot;
model.sym_u = sym_u;
model.expr_f_expl = expr_f_expl;
model.expr_f_impl = expr_f_impl;
model.h = T_INV;
model.V_x = V_x;
model.V_u = V_u;
model.V_x_e = V_x_e;
model.W = W;
model.W_e = W_e;
model.y_ref = y_ref;
model.y_ref_e = y_ref_e;
model.x0 = x0;
model.Jbu = Jbu;
model.lbu = lbu;
model.ubu = ubu;



return;

