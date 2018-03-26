function [E_FACT,E_FACT2, E_Lambda, Lowerbound, model]=VB_CP_ALS(X,D,constraints,varargin)
%VB_CP_ALS Fits a Candecomp/PARAFAC (CP) tensor decomposition model to "X"
%with "D" components/rank and subject to the supplied "constraints".
% INPUTS:
% There are 3 required inputs:
%   X:              A tensor (2nd order or higher) with the observed data.
%                       Missing values should be represented by NaN.
%   D:              The number of components or rank to be used.
%   constraints:    A cell(ndims(X),1) containing a string which specify
%                   the constraint on the corresponding mode. Each factor
%                   follows a truncated(0,infinity) normal (by default),
%                   but can be changed to an exponential distribution by
%                   adding the prefix 'exponential ' to the constraint.
%                   Valid constraints are:
%       'constant'    Fixed precision/rate of the distribution.
%       'scale'       Same precision/rate for all elements.
%       'ard'         Same precision/rate for each column (Automatic
%                       relevance Determination).
%       'sparse'      Individual precision/rate for each element (Sparsity). 
%       'infty'       The factor follows a Uniform(0,1) distribution (Cannot
%                       be combined with exponential).
%
%       'scale-shared'  Same as 'scale' but shared between multiple modes.     
%       'ard-shared'    Same as 'ard' but shared between multiple modes.   
%       'sparse-shared' Same as 'sparse' but shared between multiple modes.
%                         For sparsity, the number of observations in the
%                         shared modes must be equal.
%   varargin:   Passing optional input arguments
%      'conv_crit'      Convergence criteria, the algorithm stops if the
%                       relative change in lowerbound is below this value
%                       (default 1e-8). 
%      'maxiter'        Maximum number of iteration, stops here if
%                       convergence is not achieved (default 500).
%      'model_tau'      Model the precision of homoscedastic noise
%                       (default: true) 
%      'model_lambda'   Model the hyper-prior distributions on the factors
%                       (default: true) 
%      'fixed_tau'      Number of iterations before modeling noise
%                       commences (default: 10). 
%      'fixed_lambda'   Number of iterations before modeling hyper-prior
%                       distributions (default: 5).
%      'inference'      Defines how probabilistic inference should be
%                       performed (default: 'variational').
%
% OUTPUTS:
%   'E_FACT':       Expected first moment of the distributions.
%   'E_FACT2':      Expected second moment of the distributions.
%   'E_Lambda':     Expected first moment of the hyper-prior distributions.
%   'Lowerbound':   The value of the evidience lowerbound at each
%                   iteration (for variational inference).
%
%% Setup optimizer parameters
paramNames = {'conv_crit', 'maxiter', ...
    'model_tau', 'fixed_tau', 'tau_a0', 'tau_b0',...
    'model_lambda', 'fixed_lambda', 'lambda_a0', 'lambda_b0'...
    'init_method','inference'};

defaults = {1e-8, 500, ...
    true , 10  , 1e-4, 1e-4,...
    true , 5 , 1e-4, 1e-4,...
    0, 'variational'};

[conv_crit, maxiter, ...
    model_tau, fixed_tau, tau_alpha0, tau_beta0,...
    model_lambda, fixed_lambda, lambda_alpha0, lambda_beta0,...
    init_method, inference_scheme]...
    = internal.stats.parseArgs(paramNames, defaults, varargin{:});

assert(length(constraints) == ndims(X),...
    'The order of the data is not equal to the number of constraints.')

constraints = strtrim(constraints); % Remove leading and trailing whitespaces
inference_scheme = strtrim(inference_scheme);

% Determine how factor and noise inference should be performed.
[inference_scheme, noise_inference ] = processAndValidateInferenceSchemes(...
    inference_scheme, ndims(X) );

% The wait time must be lower than the maximum number of iterations
if maxiter < fixed_lambda
    fixed_lambda=max(0,maxiter-1);
end
if maxiter < fixed_tau
    fixed_tau=max(0,maxiter-1);
end
%% Initialize the factor matrices and noise
R=true-isnan(X); % Set observed (true) or not observed (false) status.
missing = ~(sum(R(:))==numel(X));
X(~R)=0;

Nx=ndims(X); %Tensor order
N=size(X); %Elements in each dimension

% Initialize factor and hyperprior distributions according to "constraints"
[factors, priors, shares_prior] = initializePriorDistributions(...
    N,D, missing, ...
    constraints, inference_scheme);

% Get relevant expected moments and setup X and R matricized for each mode.
Xm = cell(Nx,1);
Rm = cell(Nx,1);
E_FACT = cell(Nx,1);
E_FACT2 = cell(Nx,1);
E_Lambda = cell(Nx,1);

for i = 1:Nx
    fprintf(factors{i}.getSummary()); fprintf('\n')
    
    E_FACT{i} = factors{i}.factor;
    E_FACT2{i} = ones([N(i),D]);
    E_Lambda{i} = priors{i}.getExpFirstMoment();
    
    Xm{i} = matricizing(X,i);
    Rm{i} = matricizing(R,i);
end

% Initialize noise
noiseType = GammaNoise(Xm{1},Rm{1}, [],[],noise_inference);
Etau = noiseType.getExpFirstMoment();
SST = noiseType.getSST();

%% Additional initialization if any factors are infinity normed.
if any(my_contains(constraints, 'infty', 'IgnoreCase', true))
    
    infty_idx = my_contains(constraints, 'infty', 'IgnoreCase', true);
    
    % First update all non-infty factors
    for i = find(~infty_idx)
        factors{i}.updateFactor(i, Xm{i}, Rm{i}, E_FACT, E_FACT2, Etau)
        E_FACT{i} = factors{i}.getExpFirstMoment();
        E_FACT2{i} = factors{i}.getExpSecondMoment();
    end
    %Second update all infty factors
    for i = find(infty_idx)
        factors{i}.updateFactor(i, Xm{i}, Rm{i}, E_FACT, E_FACT2, Etau)
        E_FACT{i} = factors{i}.getExpFirstMoment();
        E_FACT2{i} = factors{i}.getExpSecondMoment();
    end
    
    % Also update Etau?
end

%% Setup how information should be displayed.
disp([' '])
disp(['Non-Negativity Constrained CP optimization'])
disp(['A ' num2str(D) ' component model will be fitted']);
disp([' '])
disp(['To stop algorithm press control C'])
disp([' ']);
dheader = sprintf(' %16s | %16s | %16s | %16s | %16s | %16s | %16s |','Iteration', ...
    'Cost', 'rel. \Delta cost', 'Noise (s.d.)', 'Var. Expl.', 'Time (sec.)', 'Time CPU (sec.)');
dline = repmat('------------------+',1,7);

%% Run ALS update
only_variational_inference = all(all(strcmpi('variational',inference_scheme)));

iter = 0;
old_cost = -inf;
delta_cost = inf;
Lowerbound = zeros(maxiter,1,'like', SST);
rmse = zeros(maxiter,1,'like', SST);
varexpl = zeros(maxiter,1,'like', SST);

fprintf('%s\n%s\n%s\n',dline, dheader, dline);
while delta_cost>=conv_crit && iter<maxiter || ...
        (iter <= fixed_lambda) || (iter <= fixed_tau)
    iter = iter + 1;
    time_tic_toc = tic;
    time_cpu = cputime;
    
    % Update factor matrices
    cost = 0;
    for i = 1:Nx % solve one factor at a time
        factors{i}.updateFactor(i, Xm{i}, Rm{i}, E_FACT, E_FACT2, Etau)
        E_FACT{i} = factors{i}.getExpFirstMoment();
        E_FACT2{i} = factors{i}.getExpSecondMoment();
        
        % Updates local hyperparameters
        if model_lambda && iter > fixed_lambda  && ~ shares_prior(i)
            if my_contains(constraints{i},'Exponential','IgnoreCase',true)
                factors{i}.updateFactorPrior(E_FACT{i})
            else
                factors{i}.updateFactorPrior(E_FACT2{i})
            end
            %E_Lambda{i} = factors{i}.hyperparameter.getExpFirstMoment();
            E_Lambda{i} = priors{i}.getExpFirstMoment();
        end
        
        if ~shares_prior(i)
            cost = cost + factors{i}.calcCost()...
                + priors{i}.calcCost();
            assert(isreal(cost))
        end
    end
    
    % Updates local-shared hyperparameters
    if any(shares_prior)
        for j = unique(nonzeros(shares_prior))
            shared_idx = find(shares_prior == j)';
            fsi = shared_idx(1);
            if my_contains(constraints{fsi},'Exponential','IgnoreCase',true)
                factors{fsi}.updateFactorPrior(E_FACT(shared_idx));
            else
                factors{fsi}.updateFactorPrior(E_FACT2(shared_idx));
            end
            
            cost = cost + priors{fsi}.calcCost();
            for f = shared_idx
                cost = cost+factors{f}.calcCost();
                E_Lambda{f} = priors{f}.getExpFirstMoment();
                
                assert(all(all(E_Lambda{fsi} == E_Lambda{shared_idx(f)})), 'Error, shared prior is not shared');
            end
        end
    end
    
    % Updates global hyperparameters
    % if any?
    
    % Update noise precision
    if model_tau && iter > fixed_tau
        noiseType.updateNoise(Xm{1},Rm{1}, E_FACT, E_FACT2);
        Etau = noiseType.getExpFirstMoment();
    else
        noiseType.calcSSE(Xm{1},Rm{1}, E_FACT, E_FACT2);
    end
    SSE = noiseType.getSSE();
    
    % Cost contribution from the likelihood
    cost = cost + noiseType.calcCost();
    assert(isreal(cost))
    
    delta_cost = (cost-old_cost)/abs(cost);
    
    % delta_cost is only garanteed to be positive for variational
    % inference, so for sampling, it should just keep going until the
    % change is low enough.
    if only_variational_inference
        [~, c_fixed_lambda, c_fixed_tau] = isElboDiverging(...
            iter, delta_cost, conv_crit,...
            {'hyperprior (lambda)', model_lambda, fixed_lambda},...
            {'noise (tau)', model_tau, fixed_tau});
        fixed_lambda = min(fixed_lambda, c_fixed_lambda);
        fixed_tau = min(fixed_tau, c_fixed_tau);
    else
        delta_cost = abs(delta_cost);
    end
    
    Lowerbound(iter) = cost;
    rmse(iter) = SSE;
    varexpl(iter) = 1-SSE/SST;
    old_cost = cost;
    
    %% Display
    if mod(iter,50)==0
        fprintf('%s\n%s\n%s\n',dline, dheader, dline);
    end
    %%
    if mod(iter,1)==0
        time_tic_toc = toc(time_tic_toc);
        time_cpu = cputime-time_cpu;
        
        fprintf(' %16i | %16.4e | %16.4e | %16.4e | %16.4f | %16.4f | %16.4f |\n',...
            iter, cost, delta_cost, sqrt(1./Etau), 1-SSE/SST, time_tic_toc, time_cpu);
    end
    
end
Lowerbound(iter+1:end)=[];
disp('Done John')