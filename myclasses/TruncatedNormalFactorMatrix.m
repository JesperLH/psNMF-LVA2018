classdef TruncatedNormalFactorMatrix < FactorMatrixInterface
    % TruncatedNormalFactorMatrix
    %   Detailed explanation goes here
    
    properties
        opt_options = struct('hals_updates',10,...
            'permute', true);
        lowerbound = 0;
        upperbound = inf;
        data_has_missing;
        factor_var;
        factor_mu;
        factor_sig2;
        
    end
    
    properties (Constant)
        supported_inference_methods = {'variational','sampling'}
        inference_default = 'variational';
    end
    
    properties (Access = public)%private)
        logZhatOUT;
    end
    
    %% Required functions
    methods
        function obj = TruncatedNormalFactorMatrix(...
                shape, prior_choice, has_missing,...
                inference_method)
            
            obj.hyperparameter = prior_choice;
            obj.distribution = 'Element-wise truncated normal distribution';
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
                self.hals_update(update_mode, Xm, Rm, eFact, eFact2, eNoise)
            else
                error('Unknown optimization method')
            end
            
        end
        
        function updateFactorPrior(self, eFact2)
            if iscell(eFact2)
                self.hyperparameter.updatePrior(...
                    cellfun(@(x)x ./ 2,eFact2,'UniformOutput',false));
            else
                self.hyperparameter.updatePrior(eFact2 / 2)
            end
        end
        
        function eFact = getExpFirstMoment(self)
            eFact = self.factor;
        end
        
        function eFact2 = getExpSecondMoment(self)
            eFact2=self.factor.^2+self.factor_var;
        end
        
        function cost = calcCost(self)
            if strcmpi(self.inference_method,'variational')
                cost = self.getLogPrior()+sum(sum(self.getEntropy()));
            elseif strcmpi(self.inference_method,'sampling')
                cost = self.getLogPrior();
            end
            
        end
        
        function entro = getEntropy(self)
            %Entropy calculation
            sig = sqrt(self.factor_sig2);
            alpha=(self.lowerbound-self.factor_mu)./sig;
            beta=(self.upperbound-self.factor_mu)./sig;
            
            if self.upperbound==inf
                if isa(alpha,'gpuArray')
                    r=exp(log(abs(alpha))+self.log_psi_func(alpha)-self.logZhatOUT); %GPU ready
                else
                    r=real(exp(log(alpha)+self.log_psi_func(alpha)-self.logZhatOUT));
                end
            else
                if isa(alpha,'gpuArray')
                    r=exp(log(abs(alpha))+self.log_psi_func(alpha)-self.logZhatOUT)...
                        -exp(log(abs(beta))+self.log_psi_func(beta)-self.logZhatOUT); %GPU ready
                else
                    r=real(exp(log(alpha)+self.log_psi_func(alpha)-self.logZhatOUT)...
                        -exp(log(beta)+self.log_psi_func(beta)-self.logZhatOUT));
                end
            end
            entro = log(sqrt(2*pi*exp(1)))+log(sig)+self.logZhatOUT+0.5*r;
            
            assert(~any(isnan(entro(:))),'Entropy was NaN')
            assert(~any(isinf(entro(:))),'Entropy was Inf')
        end
        
        function logp = getLogPrior(self)
            % Gets the prior contribution to the cost function.
            % Note. The hyperparameter needs to be updated before calculating
            % this.
            logp = numel(self.factor)*(-log(1/2)-1/2*log(2*pi))...
                +1/2*sum(self.hyperparameter.prior_log_value(:))...
                *numel(self.factor)/prod(self.hyperparameter.prior_size)...
                -1/2*sum(sum(bsxfun(@times, self.hyperparameter.prior_value , self.getExpSecondMoment())));%-1/2*sum(sum(self.hyperparameter.prior_value .* self.getExpSecondMoment()));
            
            if strcmpi(self.inference_method,'sampling')
                logp = numel(self.factor)*log(1/2)...
                    -sum(self.logZhatOUT(:));
                warning('Check if log prior is calculated correctly when sampling.')
            end
        end
    end
    
    %% Class specific functions
    methods %(Access = private)
        
        
        function hals_update(self, update_mode, Xm, Rm, eFact, eFact2, eNoise)
            ind=1:length(eFact);
            ind(update_mode) = [];
            
            % Calculate sufficient statistics
            kr = eFact{ind(1)};
            kr2 = eFact2{ind(1)};
            for i = ind(2:end)
                kr = krprod(eFact{i},kr);
                kr2 = krprod(eFact2{i},kr2);
            end
            
            Xmkr = Xm*kr;
            if self.data_has_missing
                % Sigma is individual (regardless of ARD) because of missing values.
                kr2_sum=Rm*kr2;
                krkr=[];
                [Rkrkr, IND] = premultiplication(Rm,kr,size(eFact{update_mode},2));
            else
                kr2_sum=sum(kr2,1);
                krkr=kr'*kr;
                Rkrkr = []; IND = [];
            end
            
            % Inference specific sufficient stats
            self.calcSufficientStats(kr2_sum, eNoise);
            
            lb=self.lowerbound*ones(self.factorsize(1),1);
            ub=self.upperbound*ones(self.factorsize(1),1);
            for rep = 1:self.opt_options.hals_updates % Number of HALS updates
                % Fixed or permute update order
                if self.opt_options.permute
                    dOrder = randperm(self.factorsize(2));
                else
                    dOrder = 1:self.factorsize(2);
                end
                
                for d = dOrder
                    not_d=1:self.factorsize(2);
                    not_d(d)=[];
                    
                    if strcmpi(self.inference_method, 'variational')
                        self.updateComponentVariational(d, not_d, lb, ub, ...
                            Xmkr, eNoise, ...
                            krkr, Rkrkr, IND)
                        
                    elseif strcmpi(self.inference_method, 'sampling')
                        self.updateComponentSampling(d, not_d, lb, ub, ...
                            Xmkr, eNoise, ...
                            krkr, Rkrkr, IND)
                        
                    end
                    
                end
            end
            
        end
        
        function calcSufficientStats(self, kr2_sum, eNoise)
            if any(strcmpi(self.inference_method,{'variational','sampling'}))
                self.factor_sig2 = 1./bsxfun(@plus, kr2_sum*eNoise, ...
                    self.hyperparameter.getExpFirstMoment(self.factorsize));
                
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
            if self.data_has_missing
                mu = (Xmkr(:,d)-(sum(Rkrkr(:,IND(d,:)).*self.factor(:,not_d),2)))...
                    *eNoise.*self.factor_sig2(:,d);
                
            else
                mu= (Xmkr(:,d)-self.factor(:,not_d)*krkr(not_d,d))...
                    *eNoise.*self.factor_sig2(:,d);
            end
            assert(~any(isinf(mu(:))),'Infinite mean value?')
            self.factor_mu(:,d) = mu;
            
            % Update Factors
            if all(size(lb) == size(self.factor_sig2(:,d)))
                [self.logZhatOUT(:,d), ~ , self.factor(:,d), self.factor_var(:,d) ] = ...
                    truncNormMoments_matrix_fun(lb, ub, mu , self.factor_sig2(:,d));
            else
                [self.logZhatOUT(:,d), ~ , self.factor(:,d), self.factor_var(:,d) ] = ...
                    truncNormMoments_matrix_fun(lb, ub, mu , repmat(self.factor_sig2(:,d),length(mu),1));
            end
        end
        
        function updateComponentSampling(self, d, not_d, lb, ub, Xmkr, eNoise, ...
                krkr, Rkrkr, IND)
            if self.data_has_missing
                mu = (Xmkr(:,d)-(sum(Rkrkr(:,IND(d,:)).*self.factor(:,not_d),2)))...
                    *eNoise.*self.factor_sig2(:,d);
            else
                mu= (Xmkr(:,d)-self.factor(:,not_d)*krkr(not_d,d))...
                    *eNoise.*self.factor_sig2(:,d);
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

