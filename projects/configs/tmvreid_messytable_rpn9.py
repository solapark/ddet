_base_ = [
#    '/home/sap/VEDet/mmlab/mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '/home/sap/VEDet/mmlab/mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

'''
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs={'project': 'mmdet3d'},
            interval=10,
        )
    ]
)
'''

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-1.0, -1.0, -.25, 1.0, 1.0, .5]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)

# For nuScenes we usually do 10-class detection
class_names = ['water1', 'water2', 'pepsi', 'coca1', 'coca2', 'coca3', 'coca4', 'tea1', 'tea2', 'yogurt', 'ramen1', 'ramen2', 'ramen3', 'ramen4', 'ramen5', 'ramen6', 'ramen7', 'juice1', 'juice2', 'can1', 'can2', 'can3', 'can4', 'can5', 'can6', 'can7', 'can8', 'can9', 'ham1', 'ham2', 'pack1', 'pack2', 'pack3', 'pack4', 'pack5', 'pack6', 'snack1', 'snack2', 'snack3', 'snack4', 'snack5', 'snack6', 'snack7', 'snack8', 'snack9', 'snack10', 'snack11', 'snack12', 'snack13', 'snack14', 'snack15', 'snack16', 'snack17', 'snack18', 'snack19', 'snack20', 'snack21', 'snack22', 'snack23', 'snack24', 'green_apple', 'red_apple', 'tangerine', 'lime', 'lemon', 'yellow_quince', 'green_quince', 'white_quince', 'fruit1', 'fruit2', 'peach', 'banana', 'fruit3', 'pineapple', 'fruit4', 'strawberry', 'cherry', 'red_pimento', 'green_pimento', 'carrot', 'cabbage1', 'cabbage2', 'eggplant', 'bread', 'baguette', 'sandwich', 'hamburger', 'hotdog', 'donuts', 'cake', 'onion', 'marshmallow', 'mooncake', 'shirimpsushi', 'sushi1', 'sushi2', 'big_spoon', 'small_spoon', 'fork', 'knife', 'big_plate', 'small_plate', 'bowl', 'white_ricebowl', 'blue_ricebowl', 'black_ricebowl', 'green_ricebowl', 'black_mug', 'gray_mug', 'pink_mug', 'green_mug', 'blue_mug', 'blue_cup', 'orange_cup', 'yellow_cup', 'big_wineglass', 'small_wineglass', 'glass1', 'glass2', 'glass3']

#input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False)
bands, max_freq = 64, 8
num_views = 3
num_classes = len(class_names)
pred_size = 10 #cx, cy, cw, w, l, h, sin_rot, cos_rot, vx, vy
#code_weights = [1.0] * 10 + [0.0] * 10 * num_views
code_weights = [0.0] * pred_size + [0.0] * pred_size * num_views
#code_weights[8] = 0.2
#code_weights[9] = 0.2
#virtual_weights = 0.2
virtual_weights = 10.
wh_weights = 5.
for i in range(1, num_views + 1):
    code_weights[i * pred_size] = virtual_weights  # x
    code_weights[i * pred_size + 1] = virtual_weights  # y
    #code_weights[i * 10 + 4] = virtual_weights  # z
    #code_weights[i * 10 + 6] = virtual_weights  # sin(yaw)
    #code_weights[i * 10 + 7] = virtual_weights  # cos(yaw)
    code_weights[i * pred_size + 2] = wh_weights  # w
    #code_weights[i * 10 + 3] = virtual_weights  # l
    code_weights[i * pred_size + 5] = wh_weights  # h
    #code_weights[i * 10 + 8] = 0.2 * virtual_weights  # vx
    #code_weights[i * 10 + 9] = 0.2 * virtual_weights  # vy
model = dict(
    #type='VEDet',
    type='TMVReid',
    use_grid_mask=True,
    #img_backbone=dict(
    #    type='VoVNetCP',
    #    spec_name='V-99-eSE',
    #    norm_eval=True,
    #    frozen_stages=-1,
    #    input_ch=3,
    #    out_features=(
    #        'stage4',
    #        'stage5',
    #    )),
    #img_neck=dict(type='CPFPN', in_channels=[768, 1024], out_channels=256, num_outs=2),
    gt_depth_sup=False,  # use cache to supervise
    pts_bbox_head=dict(
        type='TMVReidHead',
        #debug=True,
        emb_intrinsics=True,
        pred_size=pred_size,
        num_input=300,
        input_emb_size=128,
        idx_emb_size=127,
        num_classes=num_classes,
        in_channels=256,
        num_query=900,
        position_range=point_cloud_range,
        reg_hidden_dims=[512, 512],
        code_size=(num_views + 1) * 10,
        code_weights=code_weights,
        reg_channels=10,
        num_decode_views=num_views,
        with_time=False,
        det_transformer=dict(
            #type='VETransformer',
            type='TMvReidTransformer',
            det_decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                        dict(type='PETRMultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')),
            )),
        bbox_coder=dict(
            #type='NMSFreeCoder',
            #type='TMVDetNMSFreeCoder',
            type='TMVReidNMSFreeCoder',
            max_num=300,
            num_views=num_views,
            num_classes=num_classes),
        input_ray_encoding=dict(
            type='FourierMLPEncoding',
            input_channels=15,
            hidden_dims=[int(1.5 * 15 * 2 * bands)],
            embed_dim=256,
            fourier_type='linear',
            fourier_channels=15 * 2 * bands,
            max_frequency=max_freq),
        output_det_encoding=dict(
            type='FourierMLPEncoding',
            input_channels=13,
            hidden_dims=[int(1.5 * 13 * 2 * bands)],
            embed_dim=256,
            fourier_type='linear',
            fourier_channels=13 * 2 * bands,
            max_frequency=max_freq),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_visible=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=.25/10),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_reid=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_idx=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner2=dict(
                type='EpipolarAssignerMtvReid2D',
                valid_cost_thresh=.3),
            assigner=dict(
                type='HungarianAssignerMtvReid2D',
                cls_cost=dict(type='FocalLossCost', weight=0),
                reg_cost=dict(type='BBoxMtv2DL1Cost', weight=0, pred_size=pred_size, num_views=num_views),
                iou_cost=dict(type='BBoxMtv2DIoUCost', weight=0, pred_size=pred_size, num_views=num_views),  # Fake cost. This is just to make it compatible with DETR head. 
                query_cost=dict(type='QueryCost', weight=1, num_views=num_views),  
                align_with_loss=True,
                pc_range=point_cloud_range))))

dataset_type = 'CustomMessytableRpnDataset'
#data_root = 'data/nuscenes/'
data_root = 'data/Messytable/rpn/'

#file_client_args = dict(backend='disk')
#ida_aug_conf = {
    #"resize_lim": (0.94, 1.25),
#    "resize_lim": (0.3, 0.4),
    #"final_dim": (640, 1600),
#    "final_dim": (160, 400),
#    "bot_pct_lim": (0.0, 0.0),
#    "rot_lim": (0.0, 0.0),
    #"H": 900,
#    "H": 1080,
    #"W": 1600,
#    "W": 1920,
#    "rand_flip": True,
#}
#meta_keys = ('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
#             'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
#             'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow',
#             'intrinsics', 'extrinsics', 'scale_ratio', 'dec_extrinsics', 'timestamp')
meta_keys = ('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
             'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'img_norm_cfg',
             'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow',
             'intrinsics', 'extrinsics', 'scale_ratio', 'dec_extrinsics', 'timestamp', 'rpn_x1y1x2y2')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewRpnFromFiles'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadMultiviewTargets', num_views=num_views, rpn_mode=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'rpn_cxcywh', 'rpn_emb', 'rpn_prob'], meta_keys=meta_keys)
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewRpnFromFiles'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadMultiviewTargets', num_views=num_views, rpn_mode=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names),
            dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'rpn_cxcywh', 'rpn_emb', 'rpn_prob'], meta_keys=meta_keys)
        ])

]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    #workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        #ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_train.pkl',
        ann_file=data_root + 'messytable_infos_train.pkl',
        #ann_file=data_root + 'messytable_infos_debug.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        num_views=num_views,
        #modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        #box_type_3d='LiDAR'
        ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        #ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_val.pkl',
        #ann_file=data_root + 'messytable_infos_val.pkl',
        ann_file=data_root + 'messytable_infos_val.pkl',
        #ann_file=data_root + 'messytable_infos_debug.pkl',
        classes=class_names,
        num_views=num_views,
        #modality=input_modality
        #num_load=1
        ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        #ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_val.pkl',
        #ann_file=data_root + 'messytable_infos_test.pkl',
        ann_file=data_root + 'messytable_infos_test.pkl',
        #ann_file=data_root + 'messytable_infos_debug.pkl',
        classes=class_names,
        num_views=num_views,
        #modality=input_modality,
        #num_load=10,
        #start_idx=65
        )
    )

optimizer = dict(
    type='AdamW', lr=2e-4, paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    }), weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 200
save_dir = '/data3/sap/VEDet/result/tmvreid_messytable_rpn/9'
evaluation = dict(interval=10, pipeline=test_pipeline, metric=['bbox'], show=False, eval_thresh=.1, visible_thresh=.5, reid_thresh=.1, save_dir=save_dir, img_root='/data1/sap/MessyTable/images/')
#evaluation = dict(interval=2, pipeline=test_pipeline, metric=['bbox'], eval_thresh=.1, show=True, out_dir='/data3/sap/VEDet/result')
#checkpoint_config = dict(interval=24)
checkpoint_config = dict(interval=10)
find_unused_parameters = False

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
#load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
load_from = 'work_dirs/tmvreid_messytable_rpn/epoch_200.pth'
resume_from = None
