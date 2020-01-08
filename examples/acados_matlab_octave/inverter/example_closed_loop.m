%!sudo ln -f -s /usr/bin/gcc-4.9 gcc

%% test of native matlab interface
clear all

% check that env.sh has been run
env_run = getenv('ENV_RUN');
if (~strcmp(env_run, 'true'))
	disp('ERROR: env.sh has not been sourced! Before executing this example, run:');
	disp('source env.sh');
	return;
end



%% arguments
compile_mex = 'true';
codgen_model = 'true';
% simulation
sim_method = 'irk';
%sim_sens_forw = 'false';
sim_sens_forw = 'true';
sim_num_stages = 4;
sim_num_steps = 4;
% ocp
param_scheme = 'multiple_shooting_unif_grid';
ocp_N = 50;
nlp_solver = 'sqp';
%nlp_solver = 'sqp_rti';
%nlp_solver_exact_hessian = 'false';
nlp_solver_exact_hessian = 'true';
%regularize_method = 'no_regularize';
regularize_method = 'project';
%regularize_method = 'project_reduc_hess';
%regularize_method = 'mirror';
%regularize_method = 'convexify';
nlp_solver_max_iter = 8;
nlp_solver_tol_stat = 1e-4;
nlp_solver_tol_eq   = 1e-4;
nlp_solver_tol_ineq = 1e-4;
nlp_solver_tol_comp = 1e-4;
nlp_solver_ext_qp_res = 1;
nlp_solver_step_length = 1.0;
qp_solver = 'partial_condensing_hpipm';
%qp_solver = 'full_condensing_hpipm';
%qp_solver = 'full_condensing_qpoases';
qp_solver_cond_N = 5;
qp_solver_cond_ric_alg = 0;
qp_solver_ric_alg = 0;
qp_solver_warm_start = 0;
ocp_sim_method = 'irk';
%ocp_sim_method = 'irk';
ocp_sim_method_num_stages = 4;
ocp_sim_method_num_steps = 4;
cost_type = 'linear_ls';
%cost_type = 'ext_cost';
model_name = 'ocp_inverter';


%% create model entries
model = inverter_model();

nx = model.nx;
nu = model.nu;


%% acados ocp model
ocp_model = acados_ocp_model();
ocp_model.set('name', model_name);
% dims
ocp_model.set('T', ocp_N*model.h); % horison length time
ocp_model.set('dim_nx', model.nx);
ocp_model.set('dim_nu', model.nu);
if (strcmp(cost_type, 'linear_ls'))
	ocp_model.set('dim_ny', model.ny); % number of outputs in lagrange term
	ocp_model.set('dim_ny_e', model.ny_e); % number of outputs in mayer term
end
%ocp_model.set('dim_nbx', nbx);
ocp_model.set('dim_nbu', model.nbu); % number of input bounds
%ocp_model.set('dim_ng', ng);
%ocp_model.set('dim_ng_e', ng_e);
%ocp_model.set('dim_nh', nh);
%ocp_model.set('dim_nh_e', nh_e);
% symbolics
ocp_model.set('sym_x', model.sym_x);
if isfield(model, 'sym_u')
	ocp_model.set('sym_u', model.sym_u);
end
if isfield(model, 'sym_xdot')
	ocp_model.set('sym_xdot', model.sym_xdot);
end
% cost
ocp_model.set('cost_type', cost_type);
ocp_model.set('cost_type_e', cost_type);
%if (strcmp(cost_type, 'linear_ls'))
	ocp_model.set('cost_Vu', model.V_u);
	ocp_model.set('cost_Vx', model.V_x);
	ocp_model.set('cost_Vx_e', model.V_x_e);
	ocp_model.set('cost_W', model.W);
	ocp_model.set('cost_W_e', model.W_e);
	ocp_model.set('cost_y_ref', model.y_ref);
	ocp_model.set('cost_y_ref_e', model.y_ref_e);
%else % if (strcmp(cost_type, 'ext_cost'))
%	ocp_model.set('cost_expr_ext_cost', model.expr_ext_cost);
%	ocp_model.set('cost_expr_ext_cost_e', model.expr_ext_cost_e);
%end
% dynamics
if (strcmp(ocp_sim_method, 'erk'))
	ocp_model.set('dyn_type', 'explicit');
	ocp_model.set('dyn_expr_f', model.expr_f_expl);
else % irk irk_gnsf
	ocp_model.set('dyn_type', 'implicit');
	ocp_model.set('dyn_expr_f', model.expr_f_impl);
end
% constraints
ocp_model.set('constr_x0', model.x0);
%if (ng>0)
%	ocp_model.set('constr_C', C);
%	ocp_model.set('constr_D', D);
%	ocp_model.set('constr_lg', lg);
%	ocp_model.set('constr_ug', ug);
%	ocp_model.set('constr_C_e', C_e);
%	ocp_model.set('constr_lg_e', lg_e);
%	ocp_model.set('constr_ug_e', ug_e);
%elseif (nh>0)
%	ocp_model.set('constr_expr_h', model.expr_h);
%	ocp_model.set('constr_lh', lbu);
%	ocp_model.set('constr_uh', ubu);
%	ocp_model.set('constr_expr_h_e', model.expr_h_e);
%	ocp_model.set('constr_lh_e', lh_e);
%	ocp_model.set('constr_uh_e', uh_e);
%else
%	ocp_model.set('constr_Jbx', Jbx);
%	ocp_model.set('constr_lbx', lbx);
%	ocp_model.set('constr_ubx', ubx);
	ocp_model.set('constr_Jbu', model.Jbu);
	ocp_model.set('constr_lbu', model.lbu);
	ocp_model.set('constr_ubu', model.ubu);
%end

ocp_model.model_struct



%% acados ocp opts
ocp_opts = acados_ocp_opts();
ocp_opts.set('compile_interface', compile_mex);
ocp_opts.set('codgen_model', codgen_model);
ocp_opts.set('param_scheme', param_scheme);
ocp_opts.set('param_scheme_N', ocp_N);
ocp_opts.set('nlp_solver', nlp_solver);
ocp_opts.set('nlp_solver_exact_hessian', nlp_solver_exact_hessian);
ocp_opts.set('regularize_method', regularize_method);
ocp_opts.set('nlp_solver_ext_qp_res', nlp_solver_ext_qp_res);
ocp_opts.set('nlp_solver_step_length', nlp_solver_step_length);
if (strcmp(nlp_solver, 'sqp'))
	ocp_opts.set('nlp_solver_max_iter', nlp_solver_max_iter);
	ocp_opts.set('nlp_solver_tol_stat', nlp_solver_tol_stat);
	ocp_opts.set('nlp_solver_tol_eq', nlp_solver_tol_eq);
	ocp_opts.set('nlp_solver_tol_ineq', nlp_solver_tol_ineq);
	ocp_opts.set('nlp_solver_tol_comp', nlp_solver_tol_comp);
end
ocp_opts.set('qp_solver', qp_solver);
if (strcmp(qp_solver, 'partial_condensing_hpipm'))
	ocp_opts.set('qp_solver_cond_N', qp_solver_cond_N);
	ocp_opts.set('qp_solver_ric_alg', qp_solver_ric_alg);
end
ocp_opts.set('qp_solver_cond_ric_alg', qp_solver_cond_ric_alg);
ocp_opts.set('qp_solver_warm_start', qp_solver_warm_start);
ocp_opts.set('sim_method', ocp_sim_method);
ocp_opts.set('sim_method_num_stages', ocp_sim_method_num_stages);
ocp_opts.set('sim_method_num_steps', ocp_sim_method_num_steps);
if (strcmp(sim_method, 'irk_gnsf'))
	ocp_opts.set('gnsf_detect_struct', gnsf_detect_struct);
end

ocp_opts.opts_struct



%% acados ocp
% create ocp
ocp = acados_ocp(ocp_model, ocp_opts);
ocp
ocp.C_ocp
ocp.C_ocp_ext_fun



%% acados sim model
sim_model = acados_sim_model();
% dims
sim_model.set('dim_nx', nx);
sim_model.set('dim_nu', nu);
% symbolics
sim_model.set('sym_x', model.sym_x);
if isfield(model, 'sym_u')
	sim_model.set('sym_u', model.sym_u);
end
if isfield(model, 'sym_xdot')
	sim_model.set('sym_xdot', model.sym_xdot);
end
% model
sim_model.set('T', model.h);
if (strcmp(sim_method, 'erk'))
	sim_model.set('dyn_type', 'explicit');
	sim_model.set('dyn_expr_f', model.expr_f_expl);
else % irk
	sim_model.set('dyn_type', 'implicit');
	sim_model.set('dyn_expr_f', model.expr_f_impl);
end

%sim_model.model_struct



%% acados sim opts
sim_opts = acados_sim_opts();
sim_opts.set('compile_interface', compile_mex);
sim_opts.set('codgen_model', codgen_model);
sim_opts.set('num_stages', sim_num_stages);
sim_opts.set('num_steps', sim_num_steps);
sim_opts.set('method', sim_method);
sim_opts.set('sens_forw', sim_sens_forw);

%sim_opts.opts_struct



%% acados sim
% create sim
sim = acados_sim(sim_model, sim_opts);
%sim
%sim.C_sim
%sim.C_sim_ext_fun



%% closed loop simulation
n_sim = 200;
%n_sim = 10;
x_sim = zeros(nx, n_sim+1);
x_sim(:,1) = model.x0; % initial state
u_sim = zeros(nu, n_sim);

% set trajectory initialization
%x_traj_init = zeros(nx, ocp_N+1);
%for ii=1:ocp_N x_traj_init(:,ii) = [0; pi; 0; 0]; end
x_traj_init = repmat(model.x0, 1, ocp_N+1);

u_traj_init = zeros(nu, ocp_N);
pi_traj_init = zeros(nx, ocp_N);



tic;

for ii=1:n_sim

	% set x0
	ocp.set('constr_x0', x_sim(:,ii));

	% set trajectory initialization (if not, set internally using previous solution)
	ocp.set('init_x', x_traj_init);
	ocp.set('init_u', u_traj_init);
	ocp.set('init_pi', pi_traj_init);

	% solve OCP
	ocp.solve();

	if 1
		status = ocp.get('status');
		sqp_iter = ocp.get('sqp_iter');
		time_tot = ocp.get('time_tot');
		time_lin = ocp.get('time_lin');
		time_qp_sol = ocp.get('time_qp_sol');

		fprintf('\nstatus = %d, sqp_iter = %d, time_int = %f [ms] (time_lin = %f [ms], time_qp_sol = %f [ms])\n', status, sqp_iter, time_tot*1e3, time_lin*1e3, time_qp_sol*1e3);
	end

	ocp.print('stat');

	% get solution for initialization of next NLP
	x_traj = ocp.get('x');
	u_traj = ocp.get('u');
	pi_traj = ocp.get('pi');

	% shift trajectory for initialization
	x_traj_init = [x_traj(:,2:end), x_traj(:,end)];
	u_traj_init = [u_traj(:,2:end), u_traj(:,end)];
	pi_traj_init = [pi_traj(:,2:end), pi_traj(:,end)];

    if ii>700
        figure(1)
        subplot(2,1,1);
        plot(0:ocp_N, x_traj);
        legend('iB','vDC','iM','vC','iL','vG');
        ylim([-100,200])
        subplot(2,1,2);
        plot(0:ocp_N-1, u_traj);
        legend('u1','u2');
        ylim([0,1])
    end
    
	% get solution for sim
	u_sim(:,ii) = ocp.get('u', 0);

	% set initial state of sim
	sim.set('x', x_sim(:,ii));
	% set input in sim
	sim.set('u', u_sim(:,ii));

	% simulate state
	sim.solve();

	% get new state
	x_sim(:,ii+1) = sim.get('xn');

end

avg_time_solve = toc/n_sim


% steady state solution
x_ss = x_sim(:,n_sim)
u_ss = u_sim(:,n_sim)

A = sim.get('Sx')
B = sim.get('Su')

%Q = 0.1*eye(6);
%Q(1:5,1:5) = model.W(1:5,1:5);
%R = model.W(6:7,6:7);

%P = Q;
%for ii=1:1000
%	P = Q + A'*P*A - A'*P*B*inv(R+B'*P*B)*B'*P*A;
%end
%P

% figures

%for ii=1:n_sim+1
%	x_cur = x_sim(:,ii);
%	visualize;
%end



figure(2);
subplot(2,1,1);
plot(0:n_sim, x_sim);
xlim([0 n_sim]);
legend('iB','vDC','iM','vC','iL','vG');
subplot(2,1,2);
plot(0:n_sim-1, u_sim);
xlim([0 n_sim]);
legend('uB','uINV');



status = ocp.get('status');

if status==0
	fprintf('\nsuccess!\n\n');
else
	fprintf('\nsolution failed!\n\n');
end


if is_octave()
	waitforbuttonpress;
end


return;



