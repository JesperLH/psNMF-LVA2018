classdef ExponentialFactorMatrix < TruncatedNormalFactorMatrix
    % TruncatedNormalFactorMatrix
    %   Detailed explanation goes here
    
    properties
        %        opt_options = struct('hals_updates',10,...
        %                             'permute', true);
        %         lowerbound = 0;
        %         upperbound = inf;
        %         data_has_missing;
        %         factor_var;
        %         factor_mu;
        %         factor_sig2;
        
    end
    
    properties (Constant)
        %supported_inference_methods = {'variational','sampling'}
        %inference_default = 'variational';
    end
    
    properties (Access = public)%private)
        %logZhatOUT;
    end
    
    %% Required functions
    methods
        function obj = ExponentialFactorMatrix(...
                shape, prior_choice, has_missing,...
                inference_method)
            % Call base constructer
            obj = obj@TruncatedNormalFactorMatrix(...
                shape, prior_choice, ...
                has_missing,inference_method);
            
            obj.hyperparameter = prior_choice;
            obj.distribution = 'Element-wise Exponential Distribution';
            obj.factorsize = shape;
            obj.optimization_method = 'hals';
            obj.data_has_missing = has_missing;
            
            
            if exist('inference_method','var')
                obj.inference_method = inference_method;
            else
                obj.inference_method = obj.inference_default;
            end
            
            
        end
        
        function updateFactor(self, update_mode, Xm, Rm, eFact, eFact2, eNoise)
            
            if strcmpi(self.optimization_method,'hals')
                %hals_update@TruncatedNormalFactorMatrix(self,update_mode, Xm, Rm, eFact, eFact2, eNoise);
                self.hals_update(update_mode, Xm, Rm, eFact, eFact2, eNoise)
            else
                error('Unknown optimization method')
            end
            
        end
        
        function updateFactorPrior(self, eFact2)
            self.hyperparameter.updatePrior(eFact2)
        end
        
        %function entro = getEntropy(self)
        %%% Entroy remains the same as in TruncatedNormalFactorMatrix,
        %%% since the Q-distribution is TruncatedNormal due to the squared
        %%% loss function.
        %end
        
        function logp = getLogPrior(self)
            % Gets the prior contribution to the cost function.
            % Note. The hyperparameter needs to be updated before calculating
            % this.
            
            logp = sum(sum(self.hyperparameter.getExpLogMoment(self.factorsize)))...
                - sum(sum(bsxfun(@times, self.hyperparameter.prior_value ,...
                self.getExpFirstMoment())));
            
            if strcmpi(self.inference_method,'sampling')
                warning('Check if log prior for exponential distribution, is equal when sampling and using vb')
            end
        end
    end
    
    %% Class specific functions
    methods % (Access = private)
        
        function calcSufficientStats(self, kr2_sum, eNoise)
            if any(strcmpi(self.inference_method,{'variational','sampling'}))
                %self.factor_sig2 = 1./bsxfun(@plus, kr2_sum*eNoise, ...
                %        self.hyperparameter.getExpFirstMoment(self.factorsize));
                self.factor_sig2 = 1./bsxfun(@plus, kr2_sum*eNoise, ...
                    zeros(self.factorsize));
                
                assert(~any(isinf(self.factor_sig2(:))),'Infinite sigma value?') % Sanity check
                assert(all(self.factor_sig2(:)>=0),'Variance was negative!')
            else
                
            end
        end
        
        function result = log_psi_func(self,t)
            result = -0.5*t.^2-0.5*log(2*pi);
        end
        
        
        function updateComponentVariational(self,d, not_d, lb, ub, Xmkr, eNoise, ...
                krkr, Rkrkr, IND)
            lambda = self.hyperparameter.getExpFirstMoment(self.factorsize);
            if self.data_has_missing
                mu = ((Xmkr(:,d)-(sum(Rkrkr(:,IND(d,:)).*self.factor(:,not_d),2))...
                    )*eNoise-lambda(:,d)).*self.factor_sig2(:,d);
                
            else
                mu= ((Xmkr(:,d)-self.factor(:,not_d)*krkr(not_d,d))*eNoise...
                    -lambda(:,d)).*self.factor_sig2(:,d);
            end
            assert(~any(isinf(mu(:))),'Infinite mean value?')
            self.factor_mu(:,d) = mu;
            %fprintf('(min,max) af mu = (%2.4e, %2.4e)\n',min(mu),max(mu))
            
            assert(all(self.factor_var(:)>=0))
            % Update Factors
            if all(size(lb) == size(self.factor_sig2(:,d)))
                [self.logZhatOUT(:,d), ~ , self.factor(:,d), self.factor_var(:,d) ] = ...
                    truncNormMoments_matrix_fun(lb, ub, mu , self.factor_sig2(:,d));
            else
                [self.logZhatOUT(:,d), ~ , self.factor(:,d), self.factor_var(:,d) ] = ...
                    truncNormMoments_matrix_fun(lb, ub, mu , repmat(self.factor_sig2(:,d),length(mu),1));
            end
            assert(all(self.factor_var(:)>=0))
        end
        
        function updateComponentSampling(self, d, not_d, lb, ub, Xmkr, eNoise, ...
                krkr, Rkrkr, IND)
            lambda = self.hyperparameter.getExpFirstMoment(self.factorsize);
            if self.data_has_missing
                mu = ((Xmkr(:,d)-(sum(Rkrkr(:,IND(d,:)).*self.factor(:,not_d),2))...
                    )*eNoise-lambda(:,d)).*self.factor_sig2(:,d);
                
            else
                mu= ((Xmkr(:,d)-self.factor(:,not_d)*krkr(not_d,d))*eNoise...
                    -lambda(:,d)).*self.factor_sig2(:,d);
            end
            assert(~any(isinf(mu(:))),'Infinite mean value?')
            self.factor_mu(:,d) = mu;
            
            sig = sqrt(self.factor_sig2(:,d));
            self.factor(:,d) = trandn( (lb-mu)./sig, (ub-mu)./sig ).*sig+mu;
            
            self.factor_var = 0; % The variance of 1 sample is zero
            self.logZhatOUT(:,d) = truncNorm_logZ(lb, ub, mu , self.factor_sig2(:,d));
        end
        
        
        
    end
end

