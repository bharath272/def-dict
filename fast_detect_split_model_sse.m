function [boxes, indices, pyra, allscores]=fast_detect_split_model(img, model, thresh, bbox, overlap)
if(~exist('thresh', 'var'))
	thresh=model.thresh;
end
latent=exist('bbox', 'var');

boxes=zeros(100000,11);
indices=zeros(100000,5);
cnt=0;

%compute feature pyramid
pyra=simple_featpyramid(double(img), model);
%get all the filters in a nice cell array
coarseorfine=[];
cid=[];
incid=[];
filters={};
for k=1:numel(model.components)
	coarseorfine(end+1)=0;
	cid(end+1)=k;
	incid(end+1)=0;
	filters{end+1}=model.components(k).rootfilter.w;


	F=model.components(k).finefilter;
	coarseorfine=[coarseorfine ones(1, numel(F.w))];
	cid=[cid k*ones(1, numel(F.w))];
	incid=[incid [1:numel(F.w)]];
	filters=[filters F.w(:)'];
end

%convert filters to single
for i=1:numel(filters)
	filters{i}=single(filters{i});
end

%initialize resp
for i=1:numel(pyra.feat)
	resp{i}=cell(numel(filters),1);
end
%get coarse filter responses
for i=1+model.interval:numel(pyra.feat)
	resptmp=fconv(single(pyra.feat{i}), filters(coarseorfine==0), 1, sum(coarseorfine==0));
	resp{i}(coarseorfine==0)=resptmp;
end
%get fine filter responses
for i=1:numel(pyra.feat)-model.interval
	resptmp=fconv(single(pyra.feat{i}), filters(coarseorfine==1),1,sum(coarseorfine==1));
	resp{i}(coarseorfine==1)=resptmp;
end


%get all the responses for all the levels
%for i=1:numel(pyra.feat)
%	resp{i}=fconv(pyra.feat{i},filters,1,numel(filters));
%end

%take the valid levels (for fine filter placement) : Leave out root filter octave
levels=[1:length(pyra.feat)-model.interval];


%run through levels
for l=levels

	%run through components
	for k=1:numel(model.components)
		sz=size(model.components(k).finefilter.w{1});
		szroot=size(model.components(k).rootfilter.w);

		%if this is latent variable estimation, compute overlap mask
		if(latent)
			ovmask = checkoverlap(sz,pyra,l,bbox,size(img),overlap);
			if ~any(any(ovmask)),
				continue;
        	end

		end


		%find the fine filter responses corresponding to this component at this level
		resplevel=resp{l}((coarseorfine==1) & (cid==k));
		incidscurr=incid((coarseorfine==1) & (cid==k));
		%take the max of scores
		D=model.components(k).finefilter.def;
		bestscores=resplevel{1}+D(incidscurr(1)).w;
		bestidx=ones(size(bestscores));
		
		for i=2:numel(resplevel)
			newscores=resplevel{i}+D(incidscurr(i)).w;
			bestidx(newscores>bestscores)=i;
			bestscores=max(bestscores, newscores);
		end
		%get the coarse filter response
		resproot=resp{l+model.interval}{(coarseorfine==0) & (cid==k)};
		resproot=resproot+model.components(k).rootfilter.def.w;

		%add root filter response to the appropriate fine level response
		[yidx, xidx]=ind2sub(size(bestscores), [1:numel(bestscores)]');
      	xidxcoarse=floor((xidx+1+pyra.padx)/2);
      	yidxcoarse=floor((yidx+1+pyra.pady)/2);
		bestscores(:)=bestscores(:)+resproot(sub2ind(size(resproot),yidxcoarse, xidxcoarse));
		%allscores{k}{l}=bestscores;

		% if latent, things that don't overlap go to -inf
		if(latent);
		bestscores(~ovmask)=-inf;
		end
		
		
		%find locations that score more than the threshold and get the boxes
		[I, J]=find(bestscores>thresh);

		
		if(isempty(I)) continue; end
		
		linind=sub2ind(size(bestscores),I,J);
		boxesfine=get_boxes(J,I,sz, pyra.padx, pyra.pady, pyra.scales(l));
		boxesroot=get_boxes(xidxcoarse(linind),yidxcoarse(linind),szroot, pyra.padx, pyra.pady, pyra.scales(l+model.interval));
		boxestmp=[boxesfine boxesroot];
		%add in component id and deformation id and scores
		boxestmp(:,end+1)=k;
		boxestmp(:,end+1)=incidscurr(bestidx(linind));
		boxestmp(:,end+1)=bestscores(linind);
		boxes(cnt+1:cnt+size(boxestmp,1),:)=boxestmp;

		%indices so we can get the features
		indicestmp=[J(:) I(:) xidxcoarse(linind(:)) yidxcoarse(linind(:)) l*ones(numel(I),1)];
		indices(cnt+1:cnt+size(boxestmp,1),:)=indicestmp;
		cnt=cnt+size(boxestmp,1);
	end
      
end
if(cnt==0) 
boxes=boxes([],:); indices=indices([],:);
return; end
if(latent)
	%pick the best box
	[m,i]=max(boxes(1:cnt,end));
	boxes=boxes(i,:);
	indices=indices(i,:);
else
	boxes=boxes(1:cnt,:);
	indices=indices(1:cnt,:);
end

function box=get_boxes(x,y,sz, padx, pady, scale)
x=(x-1-padx)*scale+1;
y=(y-1-pady)*scale+1;
w=sz(2)*scale;
h=sz(1)*scale;
box=[x y x+w-1 y+h-1];		
				
function ov = checkoverlap(wsz,pyra,level,bbox,imsize,overlap)
  scale = pyra.scales(level);
  padx  = pyra.padx;
  pady  = pyra.pady;
  sz = size(pyra.feat{level});
  
  %X & Y coordinates coordinates
  xmin = ([1:sz(2)-wsz(2)+1]-padx-1)*scale+1;
  ymin = ([1:sz(1)-wsz(1)+1]-pady-1)*scale+1;
  xmax=xmin+wsz(2)*scale-1;
  ymax=ymin+wsz(1)*scale-1;
  
  %clip everything, because we should only compute overlaps on visible parts
  xmin = min(imsize(2), max(1, xmin));
  ymin = min(imsize(1), max(1, ymin));
  xmax = min(imsize(2), max(1, xmax));
  ymax = min(imsize(1), max(1, ymax));

  %compute x and y intersections
  x_inter = max(min(xmax, bbox(3))-max(xmin, bbox(1))+1,0);
  y_inter = max(min(ymax, bbox(4))-max(ymin, bbox(2))+1,0);

  %full intersections
  inter=y_inter(:)*x_inter(:)';
  areas=(ymax(:)-ymin(:)+1)*(xmax(:)-xmin(:)+1)';
  ov=inter./(areas+prod(bbox(3:4)-bbox(1:2)+1)-inter);
  ov=ov>overlap;









	













