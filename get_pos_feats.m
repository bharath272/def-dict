function [posboxes, feats, featsd]=get_pos_feats(impos, model)
pixels = model.minsize * model.sbin / 2;
minsize = prod(pixels);
feats={};
featsd={};
posboxes=struct('i',{},'j',{},'bs',{},'ds', {}, 'trees',{}, 'cid',{});
allcids=unique([model.filters.blocklabel]);
for i=1:numel(impos)

  %read the image
  img=imread(impos(i).im);
  
  %if flipped, flip the image
  if(impos(i).flip) img=img(:,end:-1:1,:); end

  %get the pos boxes
  [im, boxes] = croppos(img, impos(i).boxes); 
  %prepare to detect all boxes
  [pyra, model_dp] = gdetect_pos_prepare(img, model, boxes, 0.7);

  numboxes=size(boxes,1);
  for j=1:numboxes
    fprintf('Doing %d,%d\n', i,j);
    if(impos(i).sizes(j)<minsize)
      fprintf('Skipping %d,%d: too small\n',i,j);
      continue;
    end
    bg_boxes=[1:numboxes];
    bg_boxes(j)=0;

    fg_box=j;
    %detect
    [ds, bs, trees] = gdetect_pos(pyra, model_dp, 1, j, 0.7, bg_boxes, inf);
    %if nothing came out,skip
    if(isempty(bs))
      fprintf('Skipping %d,%d: no overlap\n',i,j);
      continue;
    end
    %convert trees to struct
    treestruct = tree_mat_to_struct(trees{1});

    %assuming that this is a root-only model, the first non-terminal is the root
    idx=find([treestruct.is_leaf],1);
    %check if the double scale exists. Otherwise skip
    if(treestruct(idx).l<model.interval+1)
      fprintf('Skipping %d,%d: no double scale\n', i, j);
      continue;
    end
    %get the fmap
    fmap = pyra.feat{treestruct(idx).l};
    %get the correct filter
    filter = model.filters(model.symbols(treestruct(idx).symbol).filter);
    %use padding and filter size to crop out feats
    fsz = filter.size;
    fy = treestruct(idx).y;
    fx=treestruct(idx).x;
    f = fmap(fy:fy+fsz(1)-1, fx:fx+fsz(2)-1,:);
    %if flip, then flip
    

    %now get feats at twice the resolution. Take the box
    fx2 = 2*fx-pyra.padx-1;
    fy2 = 2*fy-pyra.pady-1;
    fsz2 = 2*fsz;
    fmap2 = pyra.feat{treestruct(idx).l-model.interval};    

    %clip box
    f2=zeros([fsz2 size(fmap,3)]);
    f2(:,:,end)=1;
    box = [fx2 fy2 fx2+fsz2(2)-1 fy2+fsz2(1)-1];
    sz = size(fmap2);
    if(box(1)<1 || box(2)<1 || box(3)>sz(2) || box(4)>sz(1))
      fprintf('Skipping %d,%d: too much outside\n',i,j);
    end
    sz = size(fmap2);
    clipped_box = max(1, min(box, sz([2 1 2 1])));
    diff=clipped_box-box;
    f2_tmp = fmap2(clipped_box(2):clipped_box(4), clipped_box(1):clipped_box(3),:);
    f2(diff(2)+(1:size(f2_tmp,1)), diff(1)+(1:size(f2_tmp,2)),:)=f2_tmp;
    feats{end+1}=f;
    featsd{end+1} = f2;

    %if filter is the flipped version, flip it
    if(filter.flip)
      feats{end} = flipfeat(feats{end});
      featsd{end}=flipfeat(featsd{end});
    end


    %store things into struct for easy reference
    posboxes(end+1).i=i;
    posboxes(end+1).j=j;
    posboxes(end+1).bs=bs;
    posboxes(end+1).ds=ds;
    posboxes(end+1).trees=trees;
    posboxes(end+1).cid = find(filter.blocklabel==allcids);

  end
end















