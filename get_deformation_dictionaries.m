function [destall, wtsall, lda_weights] = get_deformation_dictionaries(featsd, posboxes, numper, basissize, maxtrans, patchsize)
if(~exist('numper','var')) numper=10; end
if(~exist('maxtrans','var')) maxtrans=2; end
if(~exist('patchsize', 'var')) patchsize=5; end
if(~exist('basissize', 'var')) basissize=5; end

%load whitening bg
load(bg_file_name);

%next for every component
cids=unique([posboxes.cid]);
rng(1);
for k=1:numel(cids)
  disp(cids(k));
  cid=cids(k);
  F=cat(4, featsd{find([posboxes.cid]==cid)});
  sz=size(F);
  %whiten
  [R, neg] = whiten(bg, sz(2), sz(1));
  %remove trunc feats
  F=F(:,:,1:end-1,:);
  %make a large vector
  F = reshape(F, [], size(F,4));
  %whiten
  Fw = R'\(bsxfun(@minus, F, neg));
  %reshape
  Fw=reshape(Fw, sz - [0 0 1 0]);
  %get the mean
  meanFw = mean(Fw,4);
  %get flows from mean to everything
  uex=[];
  vex=[]; 
  for j=1:size(Fw,4)
    [u, v] = get_single_flow(meanFw, Fw(:,:,:,j), maxtrans, patchsize);
    uex=[uex u(:)];
    vex=[vex v(:)];
    fprintf('.');
  end
  fprintf('\n');
  %do pca
  flows=[uex; vex];
  [U,mu,vars]=pca(flows);
  [Yk, Xhat, avsq]=pcaApply(flows, U, mu, basissize);
  %cluster
  [IDX, centroids]=kmeans(Yk',numper,'emptyaction', 'singleton', 'replicates', 3000);
  %convert to permutations
  centroid_flows = bsxfun(@plus, U(:,1:basissize)*centroids', mu);
  centroid_u=centroid_flows(1:size(uex,1),:);
  centroid_v = centroid_flows(size(uex,1)+1:end,:);
  [dest, wts] = flow_to_permutation_cont(centroid_u, centroid_v, sz(1:3));
  %keyboard;
  destall{k} = dest;
  wtsall{k}=wts;
  %whiten once more the mean to get the lda model
  w = R\meanFw(:);
  w=reshape(w, sz(1:3)-[0 0 1]);
  w(:,:,end+1) = 0;
  lda_weights{k} = w;
end  








