function LVA_experiments(no_repeats, no_run, total_runs, folder_name, run_name)

if nargin < 1
    no_repeats = 1;
    no_run = 1;
    total_runs = [];
    run_name = '';
else
    mkdir(folder_name);
    run_name = [folder_name,'/',run_name];
end
datasets = {'swimmer0dB','swimmer5dB','cbcl-faces','mnist-digits'};
datasets_D = [25, 25, 7^2, 7^2];
num_iterations = 200;
datasets_maxiter = [1,1,1,1]*num_iterations;


tstart = tic;
for di = 1:length(datasets)
    %% Load data
    tic
    if strcmpi(datasets(di),'swimmer0dB')
        load('../../Data/Swimmerdataset/data.mat'); X = flipud(matricizing(Y,3)'); sx = 32; sy=32;
        %X = double(X > 1);
        %X(X==0) = 1e-4;
        X = addTensorNoise(X,0); % Noisy swimmer
        clear Y

        save_name = 'swimmer0dB';
    elseif strcmpi(datasets(di),'swimmer5dB')
        load('../../Data/Swimmerdataset/data.mat'); X = flipud(matricizing(Y,3)'); sx = 32; sy=32;
        X = addTensorNoise(X,5); % Noisy swimmer
        clear Y

        save_name = 'swimmer5dB';
    elseif strcmpi(datasets(di),'cbcl-faces')

        X = corrected_cbcldata('../../Data/cbcl-face-database/faces/train/face/');
        sx = 19;
        sy = 19;

        save_name = 'faces';
    elseif strcmpi(datasets(di),'mnist-digits')
        X = loadMNISTImages('../../Data/MNIST/train-images.idx3-ubyte');
        sx = 28;
        sy = 28;
        save_name = 'mnist';
    else
        error('Unknown dataset')
    end
    toc

    D = datasets_D(di);
    maxiter = datasets_maxiter(di);
    
    N_models = 21;


    A_comb = cell(N_models,no_repeats);
    y_lab = cell(N_models,1);
    cost_comb = zeros(N_models, no_repeats);
    time_comb = nan(N_models,2,no_repeats);
    %% Run selected models
    for m = 1:N_models
        fprintf('Running method %i of %i\n',m,N_models)
        for r = 1:no_repeats
            fprintf('\tRepeat %i of %i. ',r,no_repeats)
            t0=tic;
            tCPU = cputime;

            [model_name, A,A2, lambda, cost] = run_model(m, X, D, maxiter, datasets{di});

            time_comb(m,1,r) = toc(t0);
            time_comb(m,2,r) = cputime-tCPU;

            toc(t0)

            A_comb{m,r} = A;
            y_lab{m} = model_name;

            if isempty(cost)
                cost_comb(m,r) = norm(X-A{1}*A{2}','fro')/norm(X,'fro');
            else
                cost_comb(m,r) = cost(end);
            end
        end
    end
    %%
    if isempty(run_name) && isempty(total_runs)
        save(strcat(save_name,'.mat'), 'A_comb', 'y_lab', 'cost_comb','time_comb')
    else
       save_loc = sprintf('%s-%i-of-%i-%s.mat', run_name, no_run, total_runs,save_name);
        save(save_loc, 'A_comb', 'y_lab', 'cost_comb','time_comb',...
            'datasets', 'datasets_D', 'datasets_maxiter')
    end
        

    pause(1)
end
fprintf('Done. '); toc(tstart);

end
%%
function [model_name, A,A2, lambda, cost] = run_model(m, X, D, maxiter, dataset)
    %%% Models
    % 1) NMF - HALS
    % 2) PH - sNMF lvl 1
    % 3) PH - sNMF lvl 2
    % 4) PH - sNMF lvl 3
    % 5) PH - sNMF lvl 4
    % 6) VB NMF (shared-ard)
    % 7) YA - sNMF lvl 1
    % 8) YA - sNMF lvl 2
    % 9) YA - sNMF lvl 3
    % 10) YA - sNMF lvl 4
    % 11) VB sNMF (Inf, Sparse)
    % 12) VB sNMF (Sparse, Inf)
    % 13) VB sNMF (Sparse, I)
    % 14) VB sNMF (I, Sparse)
    % 15) VB sNMF (Sparse, Sparse)
    % 16) VB NMF (Exp)
    % 17) VB sNMF (Exp) (Inf, Sparse)
    % 18) VB sNMF (Exp) (Sparse, Inf)
    % 19) VB sNMF (Exp) (Sparse, I)
    % 20) VB sNMF (Exp) (I, Sparse)
    % 21) VB sNMF (Exp) (Sparse, Sparse)

    A2 = [];
    lambda = [];
    cost = [];
    
    
    %sparse_lvl_YA = logspace(-2,1,4);
    sparse_lvl_YA = logspace(-2,4,4);
    
    if any(strcmpi(dataset,{'swimmer0dB','swimmer5dB'}))
        sparse_lvl_PH = [0.3, 0.5, 0.7, 0.9];
    elseif strcmpi(dataset,'cbcl-faces')
        sparse_lvl_PH = [0.4, 0.6, 0.8, 0.9];
    elseif strcmpi(dataset,'mnist-digits')
        sparse_lvl_PH = [0.30, 0.48, 0.70, 0.9]; %mean digit sparsness is 0.4766
    else
        error('Unknown dataset')
    end    
    
    
    switch m
        
        case 1
            model_name = 'NMF';
            options_ncp.hals = 1;
            options_ncp.mu = 0;
            options_ncp.maxiter = maxiter;
            [~,A,~,~]=evalc('cpNonNeg(X, D, [],options_ncp);');
        case {2, 3, 4, 5}
            % Set the sparsity level
            sW = sparse_lvl_PH([2,3,4,5]==m);
            model_name = sprintf('PH-sNMF (%1.2f)',sW);
            sH = [];
            scaleX = max(X(:));
            [~,W,H] = evalc('nmfsc(X,D,sW,sH,''./nmfresults'',0, maxiter);');
            A = {sqrt(scaleX)*W; sqrt(scaleX)*H'};
            
        case 6            
            model_name = 'VB-NMF (tNormal)';
            constr = {'ard-shared','ard-shared'};
            
        case {7, 8, 9, 10}
            
            optionYAsparse.iter = maxiter; 
            optionYAsparse.eta = 1; % Fix this regularization.
            optionYAsparse.beta = sparse_lvl_YA(m==[7, 8, 9, 10]);
            optionYAsparse.dis = true;
            
            model_name = sprintf('YA-sNMF (%1.1e)',optionYAsparse.beta);
            
            [F1,F2,~,~,~]=sparsenmfnnls(X',D,optionYAsparse);
            
            A = {F2'; F1};
            
        case 11
            model_name = 'VB-sNMF (Inf, Sparse)';
            constr = {'infty', 'sparse'};
            
        case 12
            model_name = 'VB-sNMF (Sparse, Inf)';
            constr = {'sparse','infty'};
            
        case 13
            model_name = 'VB-sNMF (Sparse, I)';
            constr = {'sparse','constant'};
            
        case 14
            model_name = 'VB-sNMF (I, Sparse)';
            constr = {'constant', 'sparse'};            
            
        case 15
            model_name = 'VB-sNMF (Sparse, Sparse)';
            constr = {'sparse','sparse'};
        
        case 16            
            model_name = 'VB-NMF (Exp)';
            constr = {'exponential ard-shared','exponential ard-shared'};
        case 17
            model_name = 'VB-sNMF (Exp) (Inf, Sparse)';
            constr = {'infty', 'exponential sparse'};
            
        case 18
            model_name = 'VB-sNMF (Exp) (Sparse, Inf)';
            constr = {'exponential sparse','infty'};
            
        case 19
            model_name = 'VB-sNMF (Exp) (Sparse, I)';
            constr = {'exponential sparse','exponential constant'};
            
        case 20
            model_name = 'VB-sNMF (Exp) (I, Sparse)';
            constr = {'exponential constant', 'exponential sparse'};            
            
        case 21
            model_name = 'VB-sNMF (Exp) (Sparse, Sparse)';
            constr = {'exponential sparse','exponential sparse'};
            
        otherwise
            error('Unexpected input')
            
    end
        
    if any(m == [6,11,12,13,14,15,16:21])
         [~,A,A2,lambda,cost] = evalc(['VB_CP_ALS(X,D,constr,''maxiter'',maxiter,',...
                 '''model_lambda'',true, ''model_tau'', true, ''inference'',', ...
                 '''variational'',''fixed_lambda'',5, ''fixed_tau'',10);']);
    end
        
    

end