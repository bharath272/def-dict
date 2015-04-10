function [u,v] = get_single_flow(src, target, maxtrans, patchsize)
%get the flow from a source hog to a target hog
%all possible translations
[ti, tj] = meshgrid([-maxtrans:maxtrans],[-maxtrans:maxtrans]);

%best scores
sz=size(src);
bestscr=-inf(sz(1), sz(2));
u=zeros(sz(1), sz(2));
v=u;

I=[1:sz(1)];
J=[1:sz(2)];

h=ones(patchsize);
%for every possible translation
for k=1:numel(ti)
  %translate source by that amount
  translated_src=zeros(sz);
  I1=I + ti(k);
  J1=J + tj(k);
  
  validI = I1>=1 & I1<=sz(1);
  validJ = J1>=1 & J1<=sz(2);
  
  translated_src(I1(validI), J1(validJ),:)= src(validI, validJ, :);
  
  %take dot product
  scr=sum(translated_src.*target,3);
  
  %smooth using a box filter of patchsize
  scr = imfilter(scr, h);

  %check if resulting score is better than previous score
  valid=false(sz(1), sz(2));
  valid(I1(validI), J1(validJ))=true;
  better = scr>bestscr;
  better=better & valid;
  bestscr(better) = scr(better);
  u(better) = ti(k);
  v(better) = tj(k);
end


 
