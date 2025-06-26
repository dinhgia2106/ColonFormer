# ColonFormer configuration sử dụng mmseg framework

# Dataset settings
dataset_type = 'ColonDataset'
data_root = 'data/TrainDataset'
test_data_root = 'data/TestDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (352, 352)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(352, 352), ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(352, 352),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Model configuration
model = dict(
    type='ColonFormer',
    backbone=dict(
        type='mit_b3'
    ),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=128),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Dataset configuration
data = dict(
    samples_per_gpu=2,  # Reduced batch size for stability
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image',
        ann_dir='mask',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image',
        ann_dir='mask',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=test_data_root,
        img_dir='images',
        ann_dir='masks',
        pipeline=test_pipeline))

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-4)
optimizer_config = dict()

# Learning rate policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)

# Runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=5)
evaluation = dict(interval=5, metric='mDice')

# Logging
log_level = 'INFO'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])

# Load from checkpoint
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
work_dir = './work_dirs/colonformer_mmseg' 