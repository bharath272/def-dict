function chosen = flow_to_permutation(u, v, sz)
%converts a flow representation to a permutation representation,
%where for each feature in the source we provide an index that says
%which feature in the target it has chosen
[i, j]=ind2sub(sz(1:2), [1:prod(sz(1:2))]');
idest = max(1, min(sz(1), bsxfun(@plus,i,u)));
jdest = max(1, min(sz(2), bsxfun(@plus,j,v)));
chosen=[];

for k=1:sz(3)
  ch = my_sub2ind(sz, idest, jdest, k*ones(size(idest)));
  chosen=[chosen; ch];
end

function ind=my_sub2ind(sz, I, J, K)
ind = I + sz(1)*(J-1) +sz(1)*sz(2)*(K-1);
