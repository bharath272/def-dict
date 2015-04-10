function [ap, prec, rec, s1]=get_ap(scores, labels, areas)
if(~exist('areas', 'var'))
	areas=ones(size(labels));
end
[s1, i1]=sort(scores, 'descend');
tp=labels(i1);
fp=1-labels(i1);
tp=tp.*areas(i1);
fp=fp.*areas(i1);
tp=cumsum(tp);
fp=cumsum(fp);
prec=tp./(tp+fp);
rec=tp./sum(labels.*areas);
ap=0;
t=[0:0.01:1];
for k=1:numel(t)
	p=max(prec(rec>=t(k)));
	if(isempty(p)) p=0; end
	ap=ap+p/numel(t);
end
