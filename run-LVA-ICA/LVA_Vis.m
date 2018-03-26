%% Analyse LVA-ICA results
clear

%% Load results
folder_name = './run-LVA-ICA/results/';
n_models = 21;

%%

for dataset = {'swimmer0dB','swimmer5dB','faces', 'mnist'}
    dataset = dataset{1};
    filenames = dir([folder_name,'*test-run*',dataset,'.mat']);
    
    filenames = strcat(folder_name,{filenames.name})'
    
    no_repeats =length(filenames);
    no_methods = n_models;
    
    res_Acomb = cell(no_methods,no_repeats);
    res_cost = nan(no_methods, no_repeats);
    res_time = nan(no_methods, 2, no_repeats);
    
    rep_idx =1;
    for i = 1:length(filenames)
        load(filenames{i})
        
        for j = 1:size(A_comb,2)
            
            for m = 1:no_methods
                if m == 1
                    res_Acomb{m,rep_idx} = A_comb{m,j}';
                else
                    res_Acomb{m,rep_idx} = A_comb{m,j};
                end
                res_cost(m,rep_idx) = cost_comb(m,j);
                res_time(m, : ,rep_idx) = time_comb(m,:,j);
            end
            rep_idx = rep_idx +1;
        end
        % also load y_lab
    end
    
    [~,min_idx] = min(res_cost .* [1,1,1,1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]',[],2);
    
    A_best = cell(no_methods,1);
    best_cost = zeros(no_methods,1);
    for i = 1:no_methods
        A_best{i} = res_Acomb{i,min_idx(i)};
        best_cost(i) = res_cost(i,min_idx(i));
        y_lab{i} = strrep(y_lab{i},'VB-','p'); % Replace VB- with p for probabilistic
    end
    
    
    %%
    if strcmpi(dataset,'swimmer')
        sx = 32;
        sy = 32;
        save_loc = './run-LVA-ICA/results/analysis/';
        save_name = 'swimmer';
    
    elseif contains(dataset,'swimmer0dB')
        sx = 32;
        sy = 32;
        save_loc = './run-LVA-ICA/results/analysis/';
        save_name = 'swimmer0dB';

    elseif contains(dataset,'swimmer5dB')
        sx = 32;
        sy = 32;
        save_loc = './run-LVA-ICA/results/analysis/';
        save_name = 'swimmer5dB';

    elseif contains(dataset,'mnist')
        sx = 28;
        sy = 28;
        save_loc = './run-LVA-ICA/results/analysis/';
        save_name = 'mnist';
    elseif contains(dataset,'faces')
        sx = 19;
        sy = 19;
        save_loc = './run-LVA-ICA/results/analysis/';
        save_name = 'faces';
    end
    if ~exist(save_loc,'dir')
        mkdir(save_loc)
    end

    %%
    clear res_Acomb
    
    
    if any(contains(filenames,'swimmer'))
        % Load data and measure reconstruction error
        load('../../Data/Swimmerdataset/data.mat'); X = flipud(matricizing(Y,3)'); sx = 32; sy=32;
        if any(contains(filenames,'tripleSwim'))
            X = repmat(X,1,5);
        end
        
        best_cost_elbo = best_cost;
        for m = 1:no_methods
            best_cost(m) = norm(X-A_best{m}{1}*A_best{m}{2}','fro')/norm(X,'fro');
        end
    end
    

    
    %%
    figure('units','normalized','position',[0.1,0.1,0.63,0.95])
    %figure('units','normalized','position',[0.1,0.1,0.8,0.8])
    display_order = [1:6,16, 7:9,...
                    12,13,11,14,15,...
                    18,19,17,20, 21];
    i=1;
    while i <= length(display_order)
    %for m = 1:no_methods
        
        m = display_order(i);
        %subplot(3,5,m)
        subplot(4,5,i)
        A = A_best{m};
        if contains(y_lab{m},'YA-sNMF')
            if any(A{1}(:)<0)
                warning('A{1} contained negative values for %s',y_lab{m})
                A{1}(A{1}<0) = 0;
            end
            
            if any(A{2}(:)<0)
                warning('A{2} contained negative values for %s',y_lab{m})
                A{2}(A{2}<0) = 0;
            end
            
        end
        
        
        energy = sqrt(sum(A{1}.^2,1).*sum(A{2}.^2,1));
        [~,sort_idx]=sort(energy,'descend');
        %img = stack_components(A{1}(:,sort_idx),[sx,sy]);
        scale_by_sd = sqrt((sum(A{2}(:,sort_idx).^2,1)));
        img = stack_components(A{1}(:,sort_idx) .* scale_by_sd,[sx,sy]);
        lgrid = size(img) ./ [sx, sy]; 
        imagesc(img)
        %imshow(img, [min(img(:)), max(img(:))])
        
        colormap(flipud(gray))
        if any(contains(filenames,'faces'))
            colormap(flipud(gray))
        end
        
        grid('on'); 
        %set(gca,'XTick',0:sx:(lgrid(1)-1)*sx,'YTick',0:sy:(lgrid(2)-1)*sy, ... % Place lines
        set(gca,'XTick',(0:sx:(lgrid(1)-1)*sx)+0.5,'YTick',(0:sy:(lgrid(2)-1)*sy)+0.5, ... % Place lines
            'GridColor',[0,0,0], 'GridAlpha',0.9,... % set color
            'XTickLabel',[],'YTickLabel',[]);  % Remove tick labels
        axis image 
        
        
        
        if any(contains(filenames,'swimmer'))
            if any(m == [6,11:no_methods])
                xlabel(sprintf('ELBO=%2.1e , RMSE=%2.2f',...
                    best_cost_elbo(m), best_cost(m)),'Fontsize',12)
            else
                xlabel(sprintf('RMSE=%2.2f',best_cost(m)),'Fontsize',12)
            end
        else
            if any(m == [6,11:no_methods])
                xlabel(sprintf('ELBO = %2.2e',best_cost(m)),'Fontsize',12)
            else
                xlabel(sprintf('RMSE = %2.2f',best_cost(m)),'Fontsize',12)
            end
        end
        %colorbar
         %Slightly bigger slices (reduces white space between subplots)
        % get subplot axis position, then stretch its width and height.
        sub_pos = get(gca,'position'); 
        %set(gca,'position',sub_pos.*[1 1 1.15 1.15])
        %set(gca,'position',sub_pos.*[1 1 1.2 1.2])
        set(gca,'position',sub_pos.*[1 1 1.21 1.21])
        
        title(strrep(y_lab{m},'Sparse','S'),'Fontsize',15)    
        
        i = i+1;
    end
    print(strcat(save_loc,save_name,'-estimated_A'),'-dpng')
    print(strcat(save_loc,save_name,'-estimated_A'),'-depsc')
end

%%

return
