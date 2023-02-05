_base_ = [
    '../_base_/models/resnet34.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
dataset_type = 'CustomDataset'
classes=['daisy','dandelion','rose','sunflower','tulip']
model = dict(
	head=dict(
	num_classes=5,topk=(1,)	)
	
	)
data = dict(
	samples_per_gpu = 32,
	workers_per_gpu = 2,
    train=dict(
        type=dataset_type,
        data_prefix='data/flower_dataset/train',
        classes=classes,
	ann_file='data/flower_dataset/train.txt'
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/flower_dataset/val',
        classes=classes,
	ann_file='data/flower_dataset/val.txt'
    )
)
optimizer = dict(type='SGD',lr = 0.001,momentum=0.9,weight_decay = 0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy = 'step',step=[1])

runner = dict(type = 'EpochBasedRunner',max_epochs = 10)

load_from ='/HOME/scz0ach/run/mmclassification-master/checkpoints/resnet34_8xb32_in1k_20210831-f257d4e6.pth' 
