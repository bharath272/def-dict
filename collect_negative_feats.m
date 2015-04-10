function [feats, ids]=collect_negative_feats(imid, test, model, thresh, maxboxes)
img=imread(test(imid).im);
[boxes, indices, pyra]=fast_detect_split_model_sse(img, model, thresh);
flipimg=lrflip(img, []);
[boxes2, indices2, pyra2]=fast_detect_split_model_sse(flipimg, model, thresh);
if(size(boxes,1)+size(boxes2,1)>maxboxes)
	%too many boxes found. Collect only the highest scoring ones
	[s1, i1]=sort([boxes(:,end); boxes2(:,end)], 'descend');
	thresh=s1(maxboxes+1);
	ind1=boxes(:,end)>thresh;
	ind2=boxes2(:,end)>thresh;
	indices=indices(ind1,:);
	indices2=indices2(ind2,:);
	boxes=boxes(ind1,:);
	boxes2=boxes2(ind2,:);
end



fprintf('Found %d boxes.\n', size(boxes,1)+size(boxes2,1));

for k=1:numel(model.components)
	idx=find(boxes(:,end-2)==k);
	featind=indices(idx,:);
	idx2=find(boxes2(:,end-2)==k);
	featind2=indices2(idx2,:);
	feats{k}=[];	


	coarsenum=numel(model.components(k).rootfilter.w);
	finenum=numel(model.components(k).finefilter.w{1});
	coarsesz=size(model.components(k).rootfilter.w);
	finesz=size(model.components(k).finefilter.w{1});
	numdef=numel(model.components(k).finefilter.w);
	feats{k}=zeros(coarsenum+finenum,size(featind,1)+size(featind2,1));
	ids{k}=zeros(size(featind,1)+size(featind2,1),4);
	cnt=0;
	for l=1:size(featind,1)
		i=featind(l,:);
		feats{k}(1:coarsenum,cnt+1)=  reshape(pyra.feat{i(end)+model.interval}(i(end-1):i(end-1)+coarsesz(1)-1, i(end-2):i(end-2)+coarsesz(2)-1,:), [],1);
		feats{k}(coarsenum+1:coarsenum+finenum,cnt+1)=reshape(pyra.feat{i(end)}(i(2):i(2)+finesz(1)-1, i(1):i(1)+finesz(2)-1,:), [], 1);
		%feats{k}(coarsenum+finenum+1,cnt+1)=1;
		%feats{k}(coarsenum+finenum+1+boxes(idx(l),end-1),cnt+1)=1;
		%feats{k}(end,cnt+1)=boxes(idx(l),end);
		ids{k}(cnt+1,:)=[imid i([1 2 5])];
		cnt=cnt+1;
	end
	for l=1:size(featind2,1)
		i=featind2(l,:);
		feats{k}(1:coarsenum,cnt+1)=  reshape(pyra2.feat{i(end)+model.interval}(i(end-1):i(end-1)+coarsesz(1)-1, i(end-2):i(end-2)+coarsesz(2)-1,:), [],1);
		feats{k}(coarsenum+1:coarsenum+finenum,cnt+1)=reshape(pyra2.feat{i(end)}(i(2):i(2)+finesz(1)-1, i(1):i(1)+finesz(2)-1,:), [], 1);
		%feats{k}(coarsenum+finenum+1,cnt+1)=1;
		%feats{k}(coarsenum+finenum+1+boxes2(idx2(l),end-1),cnt+1)=1;
		%feats{k}(end,cnt+1)=boxes2(idx2(l),end);
		ids{k}(cnt+1,:)=[imid i([1 2 5])];

		cnt=cnt+1;
	end
		
end	
		



