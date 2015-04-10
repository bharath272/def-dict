function [boxes, flip]=fast_detect_split_model_flip(img, model)
if(isfield(model, 'maxstretch'))
fprintf('Spring model\n');
boxes=fast_detect_split_model_dt(img, model);
img=lrflip(img, []);
boxes2=fast_detect_split_model_dt(img, model);

else
boxes=fast_detect_split_model_sse(img, model);
img=lrflip(img, []);
boxes2=fast_detect_split_model_sse(img, model);
end

[img, boxes2]=lrflip(img, boxes2);
flip=[zeros(size(boxes,1),1); ones(size(boxes2,1),1)];
boxes=[boxes; boxes2];
