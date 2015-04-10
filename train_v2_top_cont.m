function model=train_v2_top(cls, name, cachedir, initmodel)
initmodel.interval=5;
initmodel.sbin=8;
conf=voc_config;
[pos, neg]=pascal_data(cls, conf.pascal.year);
if(isfield(pos, 'flip')) 
fprintf('Removing flipped images. Num pos before=%d\n', numel(pos));
pos=pos([pos.flip]==0); 
fprintf('Num pos after=%d\n', numel(pos));
end
cachedir=fullfile(cachedir, cls);
%name=[cls '_' name];
if(~exist(fullfile(cachedir, name, 'model200.mat')))
	model=train_v2_cont(initmodel, pos, neg(1:200), 4, 2, name, cachedir, -1-eps);
	save(fullfile(cachedir, name, 'model200.mat'), 'model');
else
	load(fullfile(cachedir, name, 'model200.mat'), 'model');
end
if(~exist(fullfile(cachedir, name, 'final_model.mat')))
	model=train_v2_cont(model, pos, neg, 1, 1, name, cachedir, -1-eps);
	save(fullfile(cachedir, name, 'final_model.mat'), 'model');
else
	load(fullfile(cachedir, name, 'final_model.mat'), 'model');
end
model.interval=10;
model.thresh=-1.05; 
