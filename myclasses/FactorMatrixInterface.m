% classdef (Attributes) ClassName
%    properties (Attributes)
%       PropertyName
%    end
%    methods (Attributes)
%       function obj = methodName(obj,arg2,...)
%          ...
%       end
%    end
%    events (Attributes)
%       EventName
%    end
%    enumeration
%       EnumName
%    end
% end
classdef (Abstract) FactorMatrixInterface < handle
    % Public properties
    
    properties (Abstract, Constant)
        supported_inference_methods;
        inference_default;
    end
    
    properties
        distribution;
        constraint;
        factor;
        factorsize;
        scale;
        hyperparameter;
        initialization;
        optimization_method;
        inference_method;
        
    end
    
    properties (Access = private)
        cost_contr = nan;
    end
    
    methods (Abstract)
        %% Fitting methods
        updateFactor(self, update_mode, Xm, Rm, eFact, eFact2, eNoise)
        updateFactorPrior(self, opt_iteration)
        
    end
    
    methods
        
        function initialize(self, init)
            if isa(init,'function_handle')
                self.factor = init(self.factorsize);
                self.initialization = func2str(init);
            elseif all(size(init) == self.factorsize)
                self.factor = init;
                self.initialization = 'Provided by user';
            else
                error('Not a valid initialization.')
            end
        end
        
        function cost_contr = getCostContribution(self)
            cost_contr = self.cost_contr;
        end
        
        function summary = getSummary(self)
            summary = sprintf('Factor Matrix with constraint... bla bla');
        end
        
        function setOptimization(self,s)
            self.inference_method = lower(s);
        end
    end
    
    %% Check validation
    methods
        function set.factorsize(self, shape)
            if length(shape) ~= 2
                error('The size of the factor must be a 2D vector.')
            elseif any(shape <= 0)
                error('The size must be positive.')
            else
                self.factorsize = shape;
            end
        end
        
        function set.inference_method(self, s)
            if any(strcmpi(s, self.supported_inference_methods))
                self.inference_method = lower(s);
            else
                warning(['Inference method "%s" is not supported.',...
                    'Using default "%s" inference instead.'],...
                    s, self.inference_default)
                self.inference_method = self.inference_default;
            end
        end
        
    end
    
    
end