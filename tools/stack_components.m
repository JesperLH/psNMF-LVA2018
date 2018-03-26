function img_stacked = stack_components(img,img_size,lgrid)
% Reshapes each colum of "img" into an image with size img_size. The
% resulting images are stacked into a grid determined by laxis.

[~,D] = size(img);
% Align into a square (or almost square) grid
if nargin < 3
    Dsq = ceil(sqrt(D)); 
    
    if D <= Dsq*(Dsq-1)
        lgrid = [Dsq-1, Dsq];
    else
        lgrid = [Dsq, Dsq];
    end
end

% Stacks the images
img_stacked = nan(lgrid.*img_size);
row = 0;
for r = 1:lgrid(1)
    for c = 1:lgrid(2)
        d = (r-1)*lgrid(2)+c;
        if d <= D
            img_stacked((1+(r-1)*img_size(1)):img_size(1)*r,...
                (1+(c-1)*img_size(2)):img_size(2)*c) = reshape(img(:,d),img_size);
        end
    end
end

%% Example with grid lines
% figure
% imagesc(img)
% [sx, sy] = img_size;
% set(gca,'XTick',(0:sx:(lgrid(1)-1)*sx)+0.5,'YTick',(0:sy:(lgrid(2)-1)*sy)+0.5, ... % Place lines
%             'GridColor',[0,0,0], 'GridAlpha',0.9,... % set color
%             'XTickLabel',[],'YTickLabel',[]);  % Remove tick labels
%axis image 

end