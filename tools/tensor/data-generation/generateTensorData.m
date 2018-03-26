function [X,FACT]=generateTensorData(Ns,D,constr,sharedARDvar)
%Ns = [6, 5, 4, 3];
%D = 2;
%constr = [0,1,2,3,5];
%   constr 5: sparse with 30% sparse
%   constr 3: unbounded in the upper limit (NO ARD)
%   constr 2: bounded with infinity norm 1
%   constr 1: sparse and unbounded in the upper limit (element-wise ARD)
%   constr 0: unbounded in the upper limit + column ARD
%X = zeros(Ns);
FACT = cell(length(Ns),1);
if nargin < 4
    sharedARDvar = (1:D)*2+1;
end
i = 1;
maxrepeats = 10;
repeats = 0;
while i <= length(Ns)
    
    if constr(i) == 0
        %
        FACT{i} = randn(Ns(i),D).^2;
    elseif constr(i) == 1
        FACT{i} = abs((randn(Ns(i),D).^3));
    elseif constr(i) == 2
        FACT{i} = randn(Ns(i),D);
        FACT{i} = FACT{i}/max(abs(FACT{i}(:)));
    elseif constr(i) == 3 ||  constr(i) == 4
        %FACT{i} = bsxfun(@times,rand(Ns(i),D).^3,sqrt(sharedARDvar));
        FACT{i} = abs(mvnrnd(zeros(D,1),sharedARDvar,Ns(i)));
    elseif constr(i) == 5
        
        FACT{i} = randn(Ns(i),D).^2;
        sparse_pattern = ones(Ns(i),D);
        for d = 1:D
            idx = randperm(Ns(i));
            sparsity_level = 0.8;
            numel_sparse = floor(Ns(i)*sparsity_level);
            sparse_pattern(idx(1:numel_sparse),d) = 0;
        end
        FACT{i} = FACT{i}.*sparse_pattern;
    else
        
        error('Constraint unspecified');
    end
    
%     if true%make_orthogonal
%         mat = zeros(size(FACT{i}));
%         idx = randperm(Ns(i));
%         for d = 1:D-1
%             mat(1+(d-1)*floor(Ns(i)/D):min(floor(Ns(i)/D)*d,Ns(i)),d) = 1;
%         end
%         mat(floor(Ns(i)/D)*d+1:end,D) = 1;
%         FACT{i} = FACT{i}.*mat;
%     end
            
    
    
    
    % Checks if the random factor has rank D, otherwise a new is generated.
    if rank(FACT{i}) == D
        i = i+1;
        repeats = 0;
    else
        repeats = repeats+1;
        if repeats > maxrepeats
            error('Could not generate the desired random factor')
        end
    end
   
    
end

%size(krprod(krprod(krprod(FACT{1},FACT{2}),FACT{3}),FACT{4}))
%size(matricizing(randn(Ns),1))
%t1=pagefun(@mtimes,gpuArray((FACT{1}(:,1)*FACT{2}(:,1)')),(permute(FACT{3}(:,1),[3,2,1])));
%size(pagefun(@mtimes,t1,permute(FACT{3},[4,3,2,1])))
if length(Ns)==2
    X = FACT{1}*FACT{2}';
else
    % CP core
    X = nmodel(FACT);
    
    % Tucker? 
%     X = FACT{1}*FACT{2}';
%     for i=3:length(Ns)
%         X=ntimes(repmat(X,[ones(1,i-1),D]),FACT{i},i,2);
%     end

end
% 
% Y=zeros(Ns);
% for r = 1:D
%     Y = Y +bsxfun(@times,FACT{1}(:,r)*FACT{2}(:,r)',permute(FACT{3}(:,r),[3,2,1]));
% end
% 
% sum(abs(X(:)-Y(:)))/sum(abs(Y(:)))
% %X-Y
% X = Y;


