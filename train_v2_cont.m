function model=train_v2(model, pos, neg, numiter, numnegiter, name, cachedir, negthresh)
%initialize
%we will allow for a total of 10000 neg features per component. 
% We will limit positive features to the first 1000, since there are few positives (change this for person)
cachedir=fullfile(cachedir, name);
if(~exist(cachedir, 'file'))
	mkdir(cachedir);
end

for k=1:numel(model.components)
	coarsefilter=model.components(k).rootfilter.w;
	finefilter=model.components(k).finefilter.w{1};
	fsz=numel(coarsefilter)+numel(finefilter);
	posfeats{k}=zeros(fsz,1000);
	negfeats{k}=zeros(fsz, 10000);
	negids{k}=zeros(10000,4); %this is so that we don't collect the same feature again and again
	numpos(k)=0;
	numneg(k)=0;
end

%max no of hard negatives to accumulate before retraining
maxneg=10000;

keep_thresh=-1.05;

for iter=1:numiter
	fprintf('Iter=%d/%d\n', iter, numiter);
	
	numpos(:)=0;
	%numneg(:)=0;
	%estimate positive latent variables
	for k=1:numel(pos)
		fprintf('Iter %d/%d : Estimating positive : %d/%d\n', iter, numiter, k, numel(pos));
		[f, cid]=collect_positive_feats(pos(k), model);
		if(isempty(cid)) continue; end
		posfeats{cid}(:,numpos(cid)+1)=f{cid};
		numpos(cid)=numpos(cid)+1;
	end			
    %keyboard;	
	%print out filter usage
	for k=1:numel(model.components)
		fprintf('Component %d got %d/%d\n', k, numpos(k), sum(numpos));
	end
	
	%Data mining
	numhardneg=0;
	for negiter=1:numnegiter
		fprintf('Neg iter=%d/%d\n', negiter, numnegiter);
		for i=1:numel(neg)
			fprintf('Iter %d/%d, Neg iter %d/%d, Doing: %d/%d\n', iter, numiter, negiter, numnegiter, i, numel(neg));

			%find hard negatives
			[f, ids] = collect_negative_feats(i,neg,model, negthresh, maxneg+1);
			for k=1:numel(model.components)
				
				%check which features we don't already have
				idx=~ismember(ids{k}, negids{k}, 'rows');
				f{k}=f{k}(:,idx);
				ids{k}=ids{k}(idx,:);

				%add them in
				negfeats{k}(:,numneg(k)+1:numneg(k)+size(f{k},2))=f{k};
				negids{k}(numneg(k)+1:numneg(k)+size(f{k},2),:)=ids{k};
				numneg(k)=numneg(k)+size(f{k},2);
				numhardneg=numhardneg+size(f{k},2);
			end
					
			
			
			%if the feature cache is full, retrain
			if(numhardneg>maxneg)
				fprintf('Retraining...\n');
				[w, newmodel, negscores] = train_latentsvm_component_cont(posfeats, negfeats, numpos, numneg, 0.005*numel(model.components), model);
				model=newmodel;
				fprintf('Number of negatives before=%d\n', sum(numneg));
				for k=1:numel(model.components)
					ind=(negscores{k}>keep_thresh);
					negfeats{k}=negfeats{k}(:,ind);
					negids{k}=negids{k}(ind,:);
					numneg(k)=sum(ind);
				end
				fprintf('Number of negatives after=%d\n', sum(numneg));
				numhardneg=0;
				
			end

				

		end
		
	end
	%before re-estimating positives, retrain once
	fprintf('Retraining...\n');
	[w, newmodel, negscores] = train_latentsvm_component_cont(posfeats, negfeats, numpos, numneg, 0.005*numel(model.components), model);
	model=newmodel;
	fprintf('Number of negatives before=%d\n', sum(numneg));
	for k=1:numel(model.components)
		ind=(negscores{k}>keep_thresh);
		negfeats{k}=negfeats{k}(:,ind);
		negids{k}=negids{k}(ind,:);
		numneg(k)=sum(ind);
	end
	fprintf('Number of negatives after=%d\n', sum(numneg));
	numhardneg=0;

	%save model
	save(fullfile(cachedir, sprintf('model_%d.mat',iter)), 'model');	
end

