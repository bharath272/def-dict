function [im,box] = lrflip(im,box)
% Assumes box is a sequence of [x1 y1 x2 y2] corner coordinates
  im  = im(:,end:-1:1,:);
  imx = size(im,2);

  if ~isempty(box),
    x1  = box(:,1);
    x2  = box(:,3);
    box(:,1) = imx - x2 + 1;
    box(:,3) = imx - x1 + 1;
  end