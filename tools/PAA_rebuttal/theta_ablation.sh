CKPT=work_dirs/tmvreid_messytable_rpn59_reproduce_261019/epoch_200.pth
DIR=projects/configs/PAA_rebuttal
for tag in \
  cls.3_vis.5_val.1 cls.5_vis.5_val.1 cls.7_vis.5_val.1 cls.9_vis.5_val.1 \
  cls.1_vis.1_val.1 cls.1_vis.3_val.1 cls.1_vis.7_val.1 cls.1_vis.9_val.1 \
  cls.1_vis.5_val.3 cls.1_vis.5_val.5 cls.1_vis.5_val.7 cls.1_vis.5_val.9
do
  for split in train test; do
    tools/dist_test.sh $DIR/MVD_CVRT_${tag}_${split}_pkl.py $CKPT 4 --save_reid_pickle
  done
done
