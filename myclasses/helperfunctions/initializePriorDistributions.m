function [factors, priors, shares_prior ] = initializePriorDistributions(...
                                        N, D, missing, ... 
                                        constraints, inference_scheme)
%INITIALIZEPRIORDISTRIBUTIONS initializes the factor and (hyper)prior
%distributions for a CP model according to the constraints and inference
%schemes. 
%   The inputs are:
%       'N'         A vector with the number of observations in each mode
%       'D'         Scalar, the number of components
%       'missing'   Are missing values present yes (true) or no (false)
%       'constraints'         
%       'inference_scheme'      variational or sampling         

                                    
factors = cell(length(N),1);
priors = cell(length(N),1);
shares_prior = zeros(length(N),1);

for i = 1:length(N)
    %% First, initialize prior on precision/rate.
    if my_contains(constraints{i}, 'exponential', 'IgnoreCase', true)
        factor_distr = 'exponential';
    else
        factor_distr = 'truncated normal';
    end
    
    if my_contains(constraints{i}, 'shared', 'IgnoreCase', true)
        % Prior is shared by multiple factor matrices, setup accordingly.
        if shares_prior(i) > 0
            % Do nothing, prior has already been initialized. 
        else
            % Initialize prior across all shared modes.
            if my_contains(constraints{i}, 'scale-shared', 'IgnoreCase', true)
                str_match = 'scale-shared';
                shared_constr = 'scale';
            elseif my_contains(constraints{i}, 'ard-shared', 'IgnoreCase', true)
                str_match = 'ard-shared';
                shared_constr = 'ard';
            elseif my_contains(constraints{i}, 'sparse-shared', 'IgnoreCase', true)
                str_match = 'sparse-shared';
                shared_constr = 'sparse';
            else
                error('Unknown constraint "%s".',constraints{i})
            end
            
            % Find the modes indices which share prior
            idx = my_contains(constraints, str_match, 'IgnoreCase', true);
            % Set shares_prior != 0, indicating which modes share prior.
            shares_prior(idx) = max(shares_prior)+1; 
            
            % Construct prior on one mode (first occurence)
            shared_idx = find(idx);
            priors{shared_idx(1)} = GammaHyperParameter(...
                                        factor_distr, shared_constr,...
                                        [N(shared_idx(1)),D], [], [],...
                                        inference_scheme{i,2});
            
            % and pass-by-reference all calls to the shared prior.
            for j = shared_idx(2:end)
                priors{j} = handle(priors{shared_idx(1)});
            end
            
            if strcmp(str_match,'sparse-shared') % Sanity check
                   assert(all(N(shared_idx(1)) == N(shared_idx(2:end))), ...
                       'Sparsity cannot be shared if the factor matrices are not of equal size!')
            end
        end
        
    % If the prior is not shared, then it is factor specific.
    elseif my_contains(constraints{i}, 'constant', 'IgnoreCase', true)
        priors{i} = HyperParameterConstant([N(i), D], 1);
    
    elseif my_contains(constraints{i}, 'scale', 'IgnoreCase', true)
        priors{i} = GammaHyperParameter(factor_distr, 'scale', [N(i),D],...
                                        [], [], inference_scheme{i,2});
    
    elseif my_contains(constraints{i}, 'ard', 'IgnoreCase', true)
        priors{i} = GammaHyperParameter(factor_distr, 'ard', [N(i),D],...
                                        [], [], inference_scheme{i,2});
    
    elseif my_contains(constraints{i}, 'sparse', 'IgnoreCase', true)
        priors{i} = GammaHyperParameter(factor_distr, 'sparse', [N(i),D],...
                                        [], [], inference_scheme{i,2});
        
    elseif my_contains(constraints{i}, 'infty', 'IgnoreCase', true)
        %No prior (or non prior)
        
    else
        error('Unknown constraint "%s".',constraints{i})
    end
    
    %% Initialize values
    if ~isempty(strfind(lower(constraints{i}),'infty'))
        factors{i} = UniformFactorMatrix([N(i), D], 0, 1, missing, inference_scheme{i,1});
        %factors{i}.initialize(@(arg) rand(arg)*2-1);
        factors{i}.initialize(@(arg) rand(arg)*1e-4);
        
        priors{i} = factors{i}.hyperparameter;
    
    elseif ~isempty(strfind(lower(constraints{i}),'exponential'))
        factors{i} = ExponentialFactorMatrix([N(i),D], handle(priors{i}), missing,...
                                             inference_scheme{i,1});
        factors{i}.initialize(@rand);
    else
        factors{i} = TruncatedNormalFactorMatrix([N(i),D], handle(priors{i}), missing,...
                                             inference_scheme{i,1});
        factors{i}.initialize(@rand);
    end

end

