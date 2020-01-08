%
% Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
% Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
% Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
% Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
%
% This file is part of acados.
%
% The 2-Clause BSD License
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.;
%

function ocp_generate_c_code(obj)
    %% check if formulation is supported
    % add checks for
    if ~strcmp( obj.model_struct.cost_type, 'linear_ls' ) || ...
        ~strcmp( obj.model_struct.cost_type_e, 'linear_ls' )
        error(['mex templating does only support linear_ls cost for now.',...
            ' Got cost_type: %s, cost_type_e: %s.\nNotice that it might still',...
            'be possible to solve the OCP from MATLAB.'], obj.model_struct.cost_type,...
            obj.model_struct.cost_type_e);
        % TODO: add
        % nonlinear least-squares
        % external cost
    elseif ~strcmp( obj.opts_struct.nlp_solver_exact_hessian, 'false')
        error(['mex templating does only support nlp_solver_exact_hessian = "false",',...
            'i.e. Gauss-Newton Hessian approximation for now.\n',...
            'Got nlp_solver_exact_hessian = "%s"\n. Notice that it might still be possible to solve the OCP from MATLAB.'],...
            obj.opts_struct.nlp_solver_exact_hessian);
        % TODO: add exact Hessian
    elseif strcmp( obj.model_struct.dyn_type, 'discrete')
        error('mex templating does only support discrete dynamics for now. Notice that it might still be possible to solve the OCP from MATLAB.');
        % TODO: implement
    elseif strcmp( obj.opts_struct.sim_method, 'irk_gnsf')
        error('mex templating does not support irk_gnsf integrator yet. Notice that it might still be possible to solve the OCP from MATLAB.');
        % TODO: implement
    end

    if ~(strcmp(obj.opts_struct.param_scheme, 'multiple_shooting_unif_grid'))
        error(['mex templating does only support uniform discretizations for shooting nodes']);
    end

    %% generate C code for CasADi functions
    % dynamics
    if (strcmp(obj.model_struct.dyn_type, 'explicit'))
        generate_c_code_explicit_ode(obj.acados_ocp_nlp_json.model);
    elseif (strcmp(obj.model_struct.dyn_type, 'implicit'))
        if (strcmp(obj.opts_struct.sim_method, 'irk'))
            opts.generate_hess = 1;
            generate_c_code_implicit_ode(...
                obj.acados_ocp_nlp_json.model, opts);
        end
    end
    
    % constraints
    if strcmp(obj.model_struct.constr_type, 'bgh') && obj.model_struct.dim_nh > 0
        generate_c_code_nonlinear_constr( obj.model_struct, obj.opts_struct,...
              fullfile(pwd, 'c_generated_code', [obj.model_struct.name '_constraints']) );
    end

    % set include and lib path
    acados_folder = getenv('ACADOS_INSTALL_DIR');
    obj.acados_ocp_nlp_json.acados_include_path = [acados_folder, '/include'];
    obj.acados_ocp_nlp_json.acados_lib_path = [acados_folder, '/lib'];

    %% remove CasADi objects from model
    model.name = obj.acados_ocp_nlp_json.model.name;
    obj.acados_ocp_nlp_json.model = model;

    %% post process numerical data (mostly cast scalars to 1-dimensional cells)
    constr = obj.acados_ocp_nlp_json.constraints;
    props = fieldnames(constr);
    for iprop = 1:length(props)
        this_prop = props{iprop};
        % add logic here if you want to work with select properties
        this_prop_value = constr.(this_prop);
        % add logic here if you want to do something based on the property's value
        if all(size(this_prop_value) == [1 1])
            constr.(this_prop) = num2cell(constr.(this_prop));
        end
    end
    obj.acados_ocp_nlp_json.constraints = constr;

    cost = obj.acados_ocp_nlp_json.cost;
    props = fieldnames(cost);
    for iprop = 1:length(props)
        this_prop = props{iprop};
        % add logic here if you want to work with select properties
        this_prop_value = cost.(this_prop);
        % add logic here if you want to do something based on the property's value
        if all(size(this_prop_value) == [1, 1])
            cost.(this_prop) = num2cell(cost.(this_prop));
        end
    end
    obj.acados_ocp_nlp_json.cost = cost;

    %% load JSON layout
    acados_folder = getenv('ACADOS_INSTALL_DIR');
    json_layout_filename = fullfile(acados_folder, 'interfaces',...
                                   'acados_template','acados_template','acados_layout.json');
    % if is_octave()
    addpath(fullfile(acados_folder, 'external', 'jsonlab'))
    acados_layout = loadjson(fileread(json_layout_filename));
    % else % Matlab
    %     acados_layout = jsondecode(fileread(json_layout_filename));
    % end

    %% reshape constraints
    dims = obj.acados_ocp_nlp_json.dims;
    constr = obj.acados_ocp_nlp_json.constraints;
    constr_layout = acados_layout.constraints;
    fields = fieldnames(constr_layout);
    for i = 1:numel(fields)
        if strcmp(constr_layout.(fields{i}){1}, 'ndarray')
            property_dim_names = constr_layout.(fields{i}){2};
            if length(property_dim_names) == 1 % vector
                this_dims = [1, dims.(property_dim_names{1})];
            else % matrix
                this_dims = [dims.(property_dim_names{1}), dims.(property_dim_names{2})];
            end
            try
                constr.(fields{i}) = reshape(constr.(fields{i}), this_dims);
            catch e
                error(['error while reshaping constr.' fields{i} ...
                    ' to dimension ' num2str(this_dims), ', got ',...
                    num2str( size(constr.(fields{i}) )) , ' .\n ',...
                    e.message ]);
            end
        end
    end
    obj.acados_ocp_nlp_json.constraints = constr;

    %% reshape cost
    cost = obj.acados_ocp_nlp_json.cost;
    cost_layout = acados_layout.cost;
    fields = fieldnames(cost_layout);
    for i = 1:numel(fields)
        if strcmp(cost_layout.(fields{i}){1}, 'ndarray')
            property_dim_names = cost_layout.(fields{i}){2};
            if length(property_dim_names) == 1 % vector
                this_dims = [1, dims.(property_dim_names{1})];
            else % matrix
                this_dims = [dims.(property_dim_names{1}), dims.(property_dim_names{2})];
            end
            try
                cost.(fields{i}) = reshape(cost.(fields{i}), this_dims);
            catch e
                    error(['error while reshaping cost.' fields{i} ...
                        ' to dimension ' num2str(this_dims), ', got ',...
                        num2str( size(cost.(fields{i}) )) , ' .\n ',...
                        e.message ]);
            end
        end
    end
    obj.acados_ocp_nlp_json.cost = cost;

    %% dump JSON file
    % if is_octave()
        % savejson does not work for classes!
        % -> consider making the acados_ocp_nlp_json properties structs directly.
        ocp_json_struct = struct(obj.acados_ocp_nlp_json);
        disable_last_warning();
        ocp_json_struct.dims = struct(ocp_json_struct.dims);
        ocp_json_struct.cost = struct(ocp_json_struct.cost);
        ocp_json_struct.constraints = struct(ocp_json_struct.constraints);
        ocp_json_struct.solver_options = struct(ocp_json_struct.solver_options);

        % remove con* fields, that are not needed in json
        ocp_json_struct = rmfield(ocp_json_struct, 'con_p_e');
        ocp_json_struct = rmfield(ocp_json_struct, 'con_h_e');
        ocp_json_struct = rmfield(ocp_json_struct, 'con_p');
        ocp_json_struct = rmfield(ocp_json_struct, 'con_h');

        json_string = savejson('',ocp_json_struct, 'ForceRootName', 0);
    % else % Matlab
    %     json_string = jsonencode(obj.acados_ocp_nlp_json);
    % end
    fid = fopen('acados_ocp_nlp.json', 'w');
    if fid == -1, error('Cannot create JSON file'); end
    fwrite(fid, json_string, 'char');
    fclose(fid);
    %% render templated code
    acados_template_mex.render_acados_templates('acados_ocp_nlp.json')
    %% compile main
    acados_template_mex.compile_main()
end
