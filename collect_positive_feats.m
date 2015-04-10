function [feats, cids, defids]=collect_positive_feats(pos, model)
for k=1:numel(model.components)
	feats{k}=[];
end

cids=[];
defids=[];
for k=1:numel(pos)
	bbox=[pos(k).x1 pos(k).y1 pos(k).x2 pos(k).y2];
	img=imread(pos(k).im);
	[img, bbox]=croppos(img, bbox);
	[flipimg, flipbbox]=lrflip(img, bbox);
	[boxes, indices, pyra]=fast_detect_split_model_sse(img, model, -inf, bbox,0.7);
	[boxesf, indicesf, pyraf]=fast_detect_split_model_sse(flipimg, model, -inf, flipbbox,0.7);
	if(isempty(boxes) && isempty(boxesf))
		continue; 
	end
	if(isempty(boxesf))
		doflip=false;
	elseif(isempty(boxes))
		doflip=true;
	else
		doflip=boxes(1,end)<boxesf(1,end);
	end
		
	if(~doflip)
		cid=boxes(1,end-2);
		defid=boxes(1,end-1);
		coarsesz=size(model.components(cid).rootfilter.w);
		finesz=size(model.components(cid).finefilter.w{1});
		numdef=numel(model.components(cid).finefilter.w);
		f1=reshape(pyra.feat{indices(end)+model.interval}(indices(4):indices(4)+coarsesz(1)-1, indices(3):indices(3)+coarsesz(2)-1,:),[],1);
		f2=reshape(pyra.feat{indices(end)}(indices(2):indices(2)+finesz(1)-1, indices(1):indices(1)+finesz(2)-1,:),[],1);
		f3=zeros(numdef,1);
		f3(defid)=1;
		f=[f1; f2]; % 1; f3; boxes(1,end)];
	else
		cid=boxesf(1,end-2);
		defid=boxesf(1,end-1);
		coarsesz=size(model.components(cid).rootfilter.w);
		finesz=size(model.components(cid).finefilter.w{1});
		numdef=numel(model.components(cid).finefilter.w);
		f1=reshape(pyraf.feat{indicesf(end)+model.interval}(indicesf(4):indicesf(4)+coarsesz(1)-1, indicesf(3):indicesf(3)+coarsesz(2)-1,:),[],1);
		f2=reshape(pyraf.feat{indicesf(end)}(indicesf(2):indicesf(2)+finesz(1)-1, indicesf(1):indicesf(1)+finesz(2)-1,:),[],1);
		f3=zeros(numdef,1);
		f3(defid)=1;
		f=[f1; f2];% 1; f3; boxesf(1,end)];

		
	end
	feats{cid}=[feats{cid} f];
	cids(k)=cid;
	defids(k)=defid;
end



function [im, box] = croppos(im, box)

% [newim, newbox] = croppos(im, box)
% Crop positive example to speed up latent search.

% crop image around bounding box
pad = 0.5*((box(3)-box(1)+1)+(box(4)-box(2)+1));
x1 = max(1, round(box(1) - pad));
y1 = max(1, round(box(2) - pad));
x2 = min(size(im, 2), round(box(3) + pad));
y2 = min(size(im, 1), round(box(4) + pad));

im = im(y1:y2, x1:x2, :);
box([1 3]) = box([1 3]) - x1 + 1;
box([2 4]) = box([2 4]) - y1 + 1;
