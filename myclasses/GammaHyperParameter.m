classdef GammaHyperParameter < HyperParameterInterface
    %UNTITLED10 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        hp_alpha = 1e-4; % Shape
        hp_beta = 1e-4; % Rate
        
        display = true;
        distr_constr;
        
    end
    
    properties (Constant)
        valid_prior_properties = {'ard','sparse','scale'};
        supported_inference_methods = {'variational','sampling'};
        inference_default = 'variational';
    end
    
    properties %(Access = private)
        est_alpha = [];
        est_beta = [];
        prior_size;
    end
    
    methods
        function obj = GammaHyperParameter(...
                factor_distribution, enforced_property, ...
                factorsize, a_shape, b_rate, inference_method)
            if strcmpi(factor_distribution,'exponential')
                obj.distr_constr = 1;
            else
                obj.distr_constr = 0.5;
            end
            
            obj.factorsize = [];
            if exist(b_rate,'var') && ~isempty(a_shape) && ~isempty(b_rate)
                obj.hp_alpha = a_shape;
                obj.hp_beta = b_rate;
            end
            
            if obj.display
                fprintf('Using default shape (%2.2e) and rate (%2.2e)\n', ...
                    obj.hp_alpha, obj.hp_beta)
            end
            
            obj.prior_property = enforced_property;
            if strcmpi(enforced_property, 'sparse')
                obj.prior_value = obj.hp_alpha/obj.hp_beta*ones(factorsize);
            elseif strcmpi(enforced_property, 'ard')
                obj.prior_value = obj.hp_alpha/obj.hp_beta*ones(1,factorsize(2));%,1);
            elseif strcmpi(enforced_property, 'scale')
                obj.prior_value = obj.hp_alpha/obj.hp_beta;
            end
            obj.prior_size = size(obj.prior_value);
            
            obj.prior_log_value = (psi(obj.hp_alpha)-log(obj.hp_beta))*ones(...
                size(obj.prior_value));
            
            if exist('inference_method','var')
                obj.inference_method = inference_method;
            else
                obj.inference_method = obj.inference_default;
            end
            
        end
        
        
        function val = getExpFirstMoment(self, factorsize)
            % Returns the expected value of the first moment (mean)
            % If factorsize is given, then the output is replicated to match.
            if nargin == 1
                val = self.prior_value;
            else
                if all(self.prior_size == factorsize)
                    val = self.prior_value;
                elseif length(self.prior_size) == 2 && self.prior_size(2) == factorsize(2)
                    val = repmat(self.prior_value, factorsize(1), 1);
                elseif self.prior_size == 1
                    val = repmat(self.prior_value, factorsize);
                else
                    error('Unknown relation between prior and factor size')
                end
            end
        end
        
        function val = getExpLogMoment(self, factorsize)
            % Returns the expected value of the first moment (mean)
            % If factorsize is given, then the output is replicated to match.
            if nargin == 1
                val = self.prior_log_value;
            else
                if all(self.prior_size == factorsize)
                    val = self.prior_log_value;
                elseif length(self.prior_size) == 2 && self.prior_size(2) == factorsize(2)
                    val = repmat(self.prior_log_value, factorsize(1), 1);
                elseif self.prior_size == 1
                    val = repmat(self.prior_log_value, factorsize);
                else
                    error('Unknown relation between prior and factor size')
                end
            end
        end
        
        function updatePrior(self, eFact2)
            
            % The same prior can be shared by multiple factor matrices
            if iscell(eFact2)
                num_shared_modes = length(eFact2);
                
                if strcmpi(self.prior_property,'sparse')
                    eFact2=sum(cat(3,eFact2{:}),3);
                else
                    eFact2 = vertcat(eFact2{:});
                end
            else
                num_shared_modes = 1;
            end
            
            [N, D] = size(eFact2);
            
            % The number of dependent parameters and update rule is
            % different for each prior property.
            if strcmpi(self.prior_property, 'scale')
                self.est_alpha = self.hp_alpha+N*D*self.distr_constr;
                self.est_beta = self.hp_beta+sum(eFact2(:));
                
            elseif strcmpi(self.prior_property, 'ard')
                self.est_alpha = self.hp_alpha+N*self.distr_constr;
                self.est_beta = self.hp_beta+sum(eFact2,1);
                
            elseif strcmpi(self.prior_property, 'sparse')
                self.est_alpha = self.hp_alpha+num_shared_modes*self.distr_constr;
                self.est_beta = self.hp_beta+eFact2;
                
            else
                error('No update rule for property "%s"',self.prior_property)
            end
            
            % Select and apply inference method
            if strcmpi(self.inference_method,'variational')
                self.prior_value = self.est_alpha./self.est_beta;
                self.prior_log_value = psi(self.est_alpha)-log(self.est_beta);
                
            elseif strcmpi(self.inference_method,'sampling')
                % Gamrnd samples from a gamma distribution with shape (alpha) and
                % scale (1./ beta)
                self.prior_value = gamrnd(repmat(self.est_alpha,self.prior_size)...
                    , 1 ./ self.est_beta);
                self.prior_log_value = log(self.prior_value);
            end
            
        end
        
        function cost = calcCost(self)
            if strcmpi(self.inference_method,'variational')
                cost = self.calcPrior()+sum(sum(self.calcEntropy()));
            elseif strcmpi(self.inference_method,'sampling')
                cost = self.calcPrior();
            end
            
        end
        
        
        function entropy_contr = calcEntropy(self)
            if isempty(self.est_beta)
                entropy_contr = numel(self.prior_value)*(...
                    -log(self.hp_beta)+self.hp_alpha...
                    -(self.hp_alpha-1)*psi(self.hp_alpha)...
                    +gammaln(self.hp_alpha));
            else
                entropy_contr =-sum(log(self.est_beta(:)))+...
                    prod(self.prior_size)*(self.est_alpha-(self.est_alpha-1)...
                    *psi(self.est_alpha)+gammaln(self.est_alpha));
            end
        end
        
        function prior_contr = calcPrior(self)
            if isempty(self.est_beta)
                prior_contr = prod(self.prior_size)*...
                    (-gammaln(self.hp_alpha)+self.hp_alpha*log(self.hp_beta))...
                    +(self.hp_alpha-1)*sum(self.prior_log_value(:))...
                    -self.hp_beta*sum(self.prior_value(:));
                
            else
                prior_contr = prod(self.prior_size)*...
                    (-gammaln(self.hp_alpha)+self.hp_alpha*log(self.hp_beta))...
                    +(self.hp_alpha-1)*sum(self.prior_log_value(:))...
                    -self.hp_beta*sum(self.prior_value(:));
            end
        end
        
        %         function set.prior_property(self, value)
        %         % Ensuring that only valid/supported prior properties can be
        %         % specified
        %            if any(strcmpi(value, self.valid_prior_properties))
        %                self.prior_property = value;
        %            else
        %                error('"%s" is not a valid/supported property.', value)
        %            end
        %         end
        
    end
end