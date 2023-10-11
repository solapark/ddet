# Nuscenes
#VEdet nuscenes train(single gpu)
python tools/train.py projects/configs/vedet_vovnet_p4_1600x640_2vview_2frame.py --work-dir work_dirs/vedet_vovnet_p4_1600x640_2vview_2frame/

#VEdet nuscenes test(single gpu)
python tools/test.py projects/configs/vedet_vovnet_p4_1600x640_2vview_2frame.py work_dirs/vedet_vovnet_p4_1600x640_2vview_2frame/latest.pth --eval bbox

#VEdet nuscenes train(multi gpu)
tools/dist_train.sh projects/configs/vedet_vovnet_p4_1600x640_2vview_2frame.py 8 --work-dir work_dirs/vedet_vovnet_p4_1600x640_2vview_2frame/

#########################################################################
# Messytable

#VEdet Messytable train
python tools/train.py projects/configs/vedet_messytable.py --work-dir work_dirs/vedet_messytable/ 

#VEdet Messytable train(multi gpu)
CUDA_VISIBLE_DEVICES=1,2 tools/dist_train.sh projects/configs/vedet_messytable.py 2 --work-dir work_dirs/vedet_messytable/

#VEdet Messytable test(multi gpu)
tools/dist_test.sh projects/configs/vedet_messytable.py work_dirs/vedet_messytable/latest.pth 8 --eval bbox

#VEdet Messytable test(single gpu)
python tools/test.py projects/configs/vedet_messytable.py work_dirs/vedet_messytable/latest.pth --eval bbox

#########################################################################
# Messytable Reid

#VEdet Messytable train
python tools/train.py projects/configs/tmvreid_messytable_rpn.py --work-dir work_dirs/tmvreid_messytable_rpn
CUDA_VISIBLE_DEVICES=1,2,3 tools/dist_train.sh projects/configs/tmvreid_messytable_rpn.py 3 --work-dir work_dirs/tmvreid_messytable_rpn/

#test
python tools/test.py projects/configs/tmvreid_messytable_rpn.py work_dirs/tmvreid_messytable_rpn/epoch_200.pth --eval bbox
CUDA_VISIBLE_DEVICES=1,2,3 tools/dist_test.sh projects/configs/tmvreid_messytable_rpn.py work_dirs/tmvreid_messytable_rpn/epoch_200.pth 3 --eval bbox
