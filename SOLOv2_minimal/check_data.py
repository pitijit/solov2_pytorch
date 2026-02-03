from data_loader.dataset import CustomIns_separate
from configs import Custom_nori_res50

cfg = Custom_nori_res50(mode='val')
dataset = CustomIns_separate(cfg)

print('Dataset length:', len(dataset))
img, labels, boxes, masks = dataset[0]

print(labels)
print(len(masks))