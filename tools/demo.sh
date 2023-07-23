#VEdet nuscenes train(single gpu)
python tools/train.py projects/configs/vedet_vovnet_p4_1600x640_2vview_2frame.py --work-dir work_dirs/vedet_vovnet_p4_1600x640_2vview_2frame/

#VEdet nuscenes test(single gpu)
python tools/test.py projects/configs/vedet_vovnet_p4_1600x640_2vview_2frame.py work_dirs/vedet_vovnet_p4_1600x640_2vview_2frame/latest.pth --eval bbox

#VEdet nuscenes train(multi gpu)
tools/dist_train.sh projects/configs/vedet_vovnet_p4_1600x640_2vview_2frame.py 8 --work-dir work_dirs/vedet_vovnet_p4_1600x640_2vview_2frame/

#VEdet Messytable train
python tools/train.py projects/configs/vedet_messytable.py --work-dir work_dirs/vedet_messytable/ 
