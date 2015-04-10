function [w, newmodel, negscores] = train_latentsvm_component(posfeats, negfeats, numpos, numneg, C, model, lbfgsopts)
%for continuous linearly itnerpolated deformations
param=get_params(model);
param.Cpos=C;
param.Cneg=C;
param.numcomp=numel(model.components)
wreg=ones(param.len,1);
for k=1:param.numcomp
	wreg(param.defroot(k))=0.01;
	wreg(param.deffine(k):param.deffine(k)+param.numdef(k)-1)=0.01;
end
param.wreg=wreg;

for k=1:param.numcomp
	posroot{k}=posfeats{k}(1:param.numroot(k),1:numpos(k));
	posfine{k}=posfeats{k}(param.numroot(k)+1:end,1:numpos(k));
	negroot{k}=negfeats{k}(1:param.numroot(k),1:numneg(k));
	negfine{k}=negfeats{k}(param.numroot(k)+1:end,1:numneg(k));
end

initw=get_w(model, param);
for k=1:param.numcomp
	[rootw, w1, compbias, defbias, dest]=get_all_templates(initw, param, k);
	[junk,poslatent{k}]=max(bsxfun(@plus, w1'*posfine{k}, defbias(:)),[],1);

end
	


% options for lbfgs
if ~exist('lbfgs_opts', 'var') || isempty(lbfgs_opts)
  lbfgs_opts.verbose = 2;
  lbfgs_opts.maxIter = 1000;
  lbfgs_opts.optTol  = 0.000001;
end

lb=-inf(size(wreg));
ub=inf(size(wreg));
% run optimizer
obj_func = @(w) obj_func(w, posroot, posfine, negroot, negfine, poslatent,param);
w = minConf_TMP(obj_func, initw, lb, ub, lbfgs_opts);

posscores=get_scores(posfeats,numpos, initw, param);
negscores=get_scores(negfeats,numneg,initw,param);
for k=1:param.numcomp
ap=get_ap([posscores{k}(:); negscores{k}(:)], [ones(numel(posscores{k}),1); zeros(numel(negscores{k}),1)]);
ap
end


%get scores
posscores=get_scores(posfeats, numpos, w, param);
negscores=get_scores(negfeats,numneg, w,param);
for k=1:param.numcomp
ap=get_ap([posscores{k}(:); negscores{k}(:)], [ones(numel(posscores{k}),1); zeros(numel(negscores{k}),1)]);
ap
end
newmodel=get_model_from_w(w, model,param);
w2=get_w(newmodel, param);
posscores=get_scores(posfeats, numpos, w2, param);
negscores=get_scores(negfeats,numneg, w2,param);
for k=1:param.numcomp
ap=get_ap([posscores{k}(:); negscores{k}(:)], [ones(numel(posscores{k}),1); zeros(numel(negscores{k}),1)]);
ap
end












function [v,g] = obj_func(w, posroot, posfine, negroot, negfine, poslatent,param)

%the number of components
numcomp=param.numcomp;

%the number of deformations
numdef=param.numdef;


g=w.*param.wreg;
v=0.5*g'*w;
loss=0;
for k=1:numcomp
	%get all the templates
	[rootw, w1, compbias, defbias, dest, wts]=get_all_templates(w, param, k);
	numdef=numel(defbias);
	

	%get all the root scores
	rootscorespos=rootw'*posroot{k};
	rootscoresneg=rootw'*negroot{k};


	%for positives, pick the right deformations and compute scores;
	scorespos = sum(posfine{k}.*w1(:,poslatent{k}),1)+defbias(poslatent{k})'+compbias+rootscorespos;

	%for negatives, find the best scoring deformations
	[scoresneg, neglatent]=max(bsxfun(@plus, w1'*negfine{k}, defbias(:)),[],1);
	scoresneg=scoresneg+compbias+rootscoresneg;

	%compute loss
	losspos=max(0,1-scorespos);
	lossneg=max(0,1+scoresneg);
	v=v+param.Cpos*sum(losspos)+param.Cneg*sum(lossneg);
		
	%pick examples with positive loss
	Ipos=(losspos>0);
	Ineg=(lossneg>0);

	%compute gradient
	start=param.rootstart(k);
	stop=start+numel(rootw)-1;

	g(start:stop)=g(start:stop)-param.Cpos*sum(posroot{k}(:,Ipos),2);
	g(start:stop)=g(start:stop)+param.Cneg*sum(negroot{k}(:,Ineg),2);
	start=param.finestart(k);
	stop=start+param.numfine(k)-1;
	
	for i=1:numdef
		for j=1:size(dest,3)
			g(start:stop)=g(start:stop) - param.Cpos*( wts(:,i,j).*sum(posfine{k}(dest(:,i,j), poslatent{k}==i & Ipos),2));
			g(start:stop)=g(start:stop) + param.Cneg*( wts(:,i,j).*sum(negfine{k}(dest(:,i,j), neglatent==i & Ineg),2));
		end 		


		%g(start:stop)=g(start:stop)-param.Cpos*sum(posfine{k}(dest(:,i),poslatent{k}==i & Ipos),2);
		%g(start:stop)=g(start:stop)+param.Cneg*sum(negfine{k}(dest(:,i),neglatent==i & Ineg),2);
		g(param.deffine(k)+i-1)=g(param.deffine(k)+i-1)-param.Cpos*sum(poslatent{k}==i & Ipos)+param.Cneg*sum(neglatent==i & Ineg);
	end
	g(param.defroot(k))=g(param.defroot(k))-param.Cpos*sum(Ipos)+param.Cneg*sum(Ineg);
	
end

function param=get_params(model)
cnt=0;
for k=1:numel(model.components)
	param.rootstart(k)=cnt+1;
	param.numroot(k)=numel(model.components(k).rootfilter.w);

	cnt=cnt+numel(model.components(k).rootfilter.w);
	param.finestart(k)=cnt+1;
	param.numfine(k)=numel(model.components(k).finefilter.origw);

	cnt=cnt+numel(model.components(k).finefilter.origw);
	param.defroot(k)=cnt+1;
	cnt=cnt+1;
	param.deffine(k)=cnt+1;
	cnt=cnt+numel(model.components(k).finefilter.w);
	param.numdef(k)=numel(model.components(k).finefilter.w);
	param.dest{k}=model.components(k).finefilter.dest;
	param.wts{k}=model.components(k).finefilter.wts;
	param.finesize_forcrop{k}=size(model.components(k).finefilter.w{1});
end
param.len=cnt;


function [rootw, w1, compbias, defbias, dest, wts]=get_all_templates(w, param, k)
rootw=w(param.rootstart(k):param.rootstart(k)+param.numroot(k)-1);
worig=w(param.finestart(k):param.finestart(k)+param.numfine(k)-1);
dest=param.dest{k};
wts=param.wts{k};

w1=zeros(prod(param.finesize_forcrop{k}), size(dest,2));
for l=1:size(dest,2)
	for i=1:size(dest,3)
		w1(:,l)=w1(:,l) + accumarray(dest(:,l,i), worig.*wts(:,l,i),[prod(param.finesize_forcrop{k}) 1]);
		
	end
end
compbias=w(param.defroot(k));
defbias=w(param.deffine(k):param.deffine(k)+param.numdef(k)-1);

	


function w=get_w(model, param)
w=zeros(param.len,1);
for k=1:numel(model.components)
	w(param.rootstart(k):param.rootstart(k)+param.numroot(k)-1)=model.components(k).rootfilter.w;
	w(param.finestart(k):param.finestart(k)+param.numfine(k)-1)=model.components(k).finefilter.origw;
	w(param.defroot(k))=model.components(k).rootfilter.def.w;
	for l=1:param.numdef(k)
		w(param.deffine(k)+l-1)=model.components(k).finefilter.def(l).w;
	end
end	


function model=get_model_from_w(w, model, param)
for k=1:numel(model.components)
	model.components(k).rootfilter.w(:)=w(param.rootstart(k):param.rootstart(k)+param.numroot(k)-1);
	model.components(k).finefilter.origw(:)=w(param.finestart(k):param.finestart(k)+param.numfine(k)-1);
	w1=model.components(k).finefilter.w;
	w2=model.components(k).finefilter.origw;

	dest=model.components(k).finefilter.dest;
	wts=model.components(k).finefilter.wts;
	for l=1:numel(w1)
		w1{l}(:)=0;
		for i=1:size(dest,3)
			w1{l}(:)=w1{l}(:)+accumarray(dest(:,l,i), w2(:).*wts(:,l,i),size(w1{l}(:)));
		end
	end
	model.components(k).finefilter.w=w1;
	
	model.components(k).rootfilter.def.w=w(param.defroot(k));
	for l=1:numel(w1)
		model.components(k).finefilter.def(l).w=w(param.deffine(k)+l-1);
	end
end


function scores=get_scores(feats, num, w, param)



for k=1:param.numcomp
	[rootw, w1, compbias, defbias, dest]=get_all_templates(w, param, k);
	featsroot=feats{k}(1:numel(rootw),1:num(k));
	featsfine=feats{k}(numel(rootw)+1:numel(rootw)+size(w1,1),1:num(k));
	scr=max(bsxfun(@plus, w1'*featsfine, defbias(:)),[],1);
	scr=scr+rootw'*featsroot+compbias;
	scores{k}=scr;

end










	



