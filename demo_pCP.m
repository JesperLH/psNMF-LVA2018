%% Demo of probabilistic CP
addpath(genpath('./'))
addpath('..?.../nway331/'); % Path to Rasmus Bro's N-way toolbox.

Ndims = [50,49,48]; D = 5;
constr = {'sparse','ard-shared','ard-shared'};
has_missing = true;

X = generateTensorData(Ndims,D,[3,3,3]);
X = addTensorNoise(X,10); % SNR = 10 dB

if has_missing
    % Create 10% missing values
    miss_idx = randperm(numel(X));
    X(miss_idx(1:round(0.10 * numel(X)))) = nan;
end

% Run pCP
[EA, ~, E_lambda, elbo] = VB_CP_ALS(X,D,constr,'maxiter',100,...
    'inference','variational');


%% Quick visualization
figure('Position',[100,500,300*length(EA),600]);
for n = 1:length(EA)
    subplot(2,length(EA),n)
    
    plot(EA{n})
    xlabel('Observations')
    ylabel('Loading')
    title(sprintf('Mode %i Factor',n))
    legend(strsplit(strtrim(sprintf('D=%i ',1:D)),' '))
    
    if ~isempty(E_lambda)
        subplot(2,length(EA),n+length(EA))
        if strcmpi(constr{n},'sparse')
            imagesc(EA{n}); colorbar;
            xlabel('Components')
            ylabel('Observations')
            title('Elementwise precision')
        elseif strcmpi(constr{n},'ard-shared')
            bar(E_lambda{n})
            xlabel('Components')
            ylabel('Precision')
            title('Columnwise precision (ARD)')
            
        end
    end
end 
