DIR=projects/configs/PAA_rebuttal
for t in .1 .3 .7 .9; do
  name=MVD_CVRT_epipolarAssignerIOU${t}
  CUDA_VISIBLE_DEVICES=4,5,6,7 tools/dist_train.sh $DIR/${name}.py 4 --work-dir work_dirs/${name}/
  CUDA_VISIBLE_DEVICES=4,5,6,7 tools/dist_test.sh $DIR/${name}.py work_dirs/${name}/epoch_200.pth 4 --save_reid_pickle
  CUDA_VISIBLE_DEVICES=4,5,6,7 tools/dist_test.sh $DIR/${name}_test_pkl.py work_dirs/${name}/epoch_200.pth 4 --save_reid_pickle
done
