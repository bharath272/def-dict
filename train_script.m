categories={'aeroplane', 'bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', 'sheep','sofa','train','tvmonitor'};

categs_to_train=[1:20];
year='2012';
cachedir='/work5/bharath2/defdict/cachedir/';
name='defdict';

conf=voc_config;
VOCinit;
ids = textread(sprintf(VOCopts.imgsetpath, 'val'), '%s');

for k=categs_to_train
  cls=categories{k};
  if(exist(fullfile(cachedir, cls, name, 'ap.mat'), 'file'))
    x1=load(fullfile(cachedir, cls, name, 'ap.mat'), 'ap');
    aps(k)=x1.ap;
    continue;
  end
  if(~exist(fullfile(cachedir, cls, name), 'file'))
    mkdir(fullfile(cachedir, cls, name));
  end


  if(exist(fullfile(cachedir, cls, name, 'posfeats.mat'), 'file'))
    x1 = load(fullfile(cachedir, cls, name, 'posfeats.mat'), 'posboxes', 'feats','featsd');
    posboxes = x1.posboxes;
    feats = x1.feats;
    featsd = x1.featsd;

  else
    %get the data
    [pos, neg, impos] = pascal_data(cls, year); 
  
    %use DPM V5 to train a root-only model
    mixmodel=pascal_train_rootmodel(cls,3);
    
    %get positive features
    [posboxes, feats, featsd]=get_pos_feats(impos, mixmodel);
    
    save(fullfile(cachedir, cls, name, 'posfeats.mat'), 'posboxes', 'feats','featsd');
  end

  if(exist(fullfile(cachedir, cls, name, 'deformations.mat'), 'file'))
    x1=load(fullfile(cachedir, cls, name, 'deformations.mat'), 'destall', 'wtsall', 'lda_weights');
    destall = x1.destall;
    wtsall = x1.wtsall;
    lda_weights = x1.lda_weights;
  else
    %get deformation dictionaries
    [destall, wtsall, lda_weights] = get_deformation_dictionaries(featsd, posboxes); 
    save(fullfile(cachedir, cls, name, 'deformations.mat'), 'destall', 'wtsall', 'lda_weights');
  end
  

  if(exist(fullfile(cachedir,cls, name, 'initmodel.mat'), 'file'))
    x1 = load(fullfile(cachedir, cls, name, 'initmodel.mat'), 'model');
    model = x1.model;
  else

    %initialize model
    model = init_model_from_def_dict(destall, wtsall, lda_weights, mixmodel);
    save(fullfile(cachedir, cls, name, 'initmodel.mat'), 'model');
  end
  %do the training
  model=train_v2_top_cont(cls, name, cachedir, model);

  %do the testing
  if(exist(fullfile(cachedir, cls, name, 'boxes.mat'), 'file'))
    x1 = load(fullfile(cachedir, cls, name, 'boxes.mat'), 'boxes');
    boxes = x1.boxes;
  else
    boxes = model_test_fast_split(name, model, ids, VOCopts);
    save(fullfile(cachedir, cls, name, 'boxes.mat'), 'boxes');
  end
  [ap, prec, recall] = pascal_eval(cls, boxes, 'val', '2012', ''); 
  save(fullfile(cachedir, cls, name, 'ap.mat'), 'ap','prec','recall');
  aps(k)=ap;

  disp(k);
end
  
   
  
