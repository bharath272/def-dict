function model = init_model_from_def_dict(dest, wts, lda_w, mixmodel)
model.components=struct('rootfilter',{},'finefilter',{});
model.maxsize=mixmodel.maxsize;
%each component corresponds to a block in the filters
filters=mixmodel.filters;
uniqueblocks=unique([mixmodel.filters.blocklabel]);
for k=1:numel(uniqueblocks)
  idx=find([filters.blocklabel]==uniqueblocks(k));
  w = mixmodel.blocks(uniqueblocks(k)).w;
  sz=mixmodel.blocks(uniqueblocks(k)).shape;
  
  %reshape 
  rootfilter.w = reshape(w, sz);
  rootfilter.def.w = 0;

  %finefilters
  finefilter=rootfilter;
  finefilter.dest=dest{k};
  finefilter.wts=wts{k};
  w1=lda_w{k};
  finefilter.origw=w1;
  finefilter.w={};
  finefilter.def.w=0;
  for l=1:size(dest{k},2)
        wtmp=zeros(size(w1));
        for i=1:size(dest{k},3)
            wtmp(:)=wtmp(:)+accumarray(dest{k}(:,l,i), w1(:).*wts{k}(:,l,i), [numel(w1) 1]);
        end
        finefilter.w{l}=wtmp;
        finefilter.def(l).w=0;
  end
  model.components(k).rootfilter=rootfilter;
  model.components(k).finefilter=finefilter;
end
  


