#VEdet nuscenes train
tools/dist_train.sh projects/configs/vedet_vovnet_p4_1600x640_2vview_2frame.py 8 --work-dir work_dirs/vedet_vovnet_p4_1600x640_2vview_2frame/

#VEdet Messytable train
python tools/train.py projects/configs/vedet_messytable.py --work-dir work_dirs/vedet_messytable/ 
