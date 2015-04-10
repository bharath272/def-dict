function [dest, wts]=flow_to_permutation_cont(u,v,sz)
%linearly interpolated version
u1=floor(u);
v1=floor(v);
u2=u1+1;
v2=v1+1;

wtsu1=u2-u; wtsu2=u-u1;
wtsv1=v2-v; wtsv2=v-v1;

dest1 = flow_to_permutation(u1,v1,sz);
wts1 = expand_wts(wtsu1.*wtsv1,sz);
dest=dest1;
wts=wts1;

dest2 = flow_to_permutation(u1, v2, sz);
wts2=expand_wts(wtsu1.*wtsv2, sz);
dest(:,:,end+1)=dest2;
wts(:,:,end+1)=wts2;

dest2 = flow_to_permutation(u2, v1, sz);
wts2=expand_wts(wtsu2.*wtsv1, sz);
dest(:,:,end+1)=dest2;
wts(:,:,end+1)=wts2;

dest2 = flow_to_permutation(u2, v2, sz);
wts2=expand_wts(wtsu2.*wtsv2, sz);
dest(:,:,end+1)=dest2;
wts(:,:,end+1)=wts2;

function wts1=expand_wts(wts, sz)
wts1=wts;
for k=2:sz(3)
    wts1=[wts1; wts];
end

