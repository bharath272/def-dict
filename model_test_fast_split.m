function res = model_test_fast_split(name,model,imnames, VOCopts)
% boxes = model_test(name,model,ims)
% Compute bounding boxes in a test set.
% boxes1 are bounding boxes from root placements
% boxes2 are bounding boxes using predictor function

res = cell(length(imnames),1);
for i = 1:length(imnames),
  fprintf('%s: testing: %d/%d\n', name, i, length(imnames));
  im = imread(sprintf(VOCopts.imgpath, imnames{i}));
  boxes = fast_detect_split_model_flip(im, model);
  if ~isempty(boxes)
    b1 = clipboxes(im, boxes, true);
   % res{i}=b1;
   %b1=boxes;
    res{i} =    nms(b1,0.5);
    
  end
  
  %showboxes(im, res{i});
end

function boxes = clipboxes(im, boxes, rootonly)
% boxes = clipboxes(im, boxes, rootonly)
% Clips boxes to image boundary.
%also removes boxes for which the root filter lies outside the image
%boundary
imy = size(im,1);
imx = size(im,2);
numparts=round((size(boxes,2)-2)/4);

%boxes(boxes(:,1)<0 & boxes(:,2)<0 & boxes(:,3)>imx+1 & boxes(:,4)>imy+1,:)=[];
for i=1:numparts
    b=boxes(:,(i-1)*4+1:i*4);
    b(:,1) = min(max(b(:,1), 1), imx);
    b(:,2) = min(max(b(:,2), 1), imy);
    b(:,3) = max(min(b(:,3), imx),1);
    b(:,4) = max(min(b(:,4), imy),1);
    boxes(:,(i-1)*4+1:i*4)=b;
end
areas=(boxes(:,4)-boxes(:,2)).*(boxes(:,3)-boxes(:,1));
boxes=boxes(areas>1e-3,:);

function [top,pick] = nms(boxes, overlap)

% [top,pick] = nms(boxes, overlap) 
% Non-maximum suppression.
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected detection.
areas=prod(boxes(:,3:4)-boxes(:,1:2)+1,2);
if isempty(boxes)
  pick = [];
else
  s = boxes(:,end);
  [vals, I] = sort(s);
  pick = [];
  while ~isempty(I)
    last = length(I);
    i    = I(last);
    pick = [pick; i];
    suppress = [last];
    bi  = boxes(i,1:end-2);
    for pos = 1:last-1
      j   = I(pos);
      bj  = boxes(j,1:end-2);
      xx1 = max(bi(1), bj(1));
      yy1 = max(bi(2), bj(2));
      xx2 = min(bi(3), bj(3));
      yy2 = min(bi(4), bj(4));
      w = xx2-xx1+1;
      h = yy2-yy1+1;
      if w > 0 && h > 0
        % compute overlap 
%         wj = bj(3)-bj(1)+1;
%         hj = bj(4)-bj(2)+1;
        o = w * h / areas(j);
        if o > overlap
          suppress = [suppress; pos];
        end
      end
    end
    I(suppress) = [];
  end  
end

top = boxes(pick,:);

