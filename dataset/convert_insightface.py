from PIL import Image
import cv2
import os
import pickle
import mxnet as mx
from tqdm import tqdm
import argparse

'''
For train dataset, insightface provide a mxnet .rec file, just install a mxnet-cpu for extract images
python convert_insightface.py --source /home/zzhuang/faces_webface_112x112 --dest /home/zzhuang/casia-webface-112x112-arcface
python convert_insightface.py --bin --source /home/zzhuang/faces_webface_112x112/agedb_30.bin --dest /home/zzhuang/arcface-test-set
'''

parser = argparse.ArgumentParser()
parser.add_argument('--source', help='source_dir')
parser.add_argument('--dest', help='dest_dir')
parser.add_argument('--bin', help='test dataset', action='store_true')


def load_mx_rec(source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(source_dir, 'train.idx'), os.path.join(source_dir, 'train.rec'),
                                           'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        label_path = os.path.join(dest_dir, str(label).zfill(6))
        os.makedirs(label_path, exist_ok=True)
        cv2.imwrite(os.path.join(label_path, str(idx).zfill(8) + '.jpg'), img)


def load_image_from_bin(bin_path, dest_dir):
    dataset_name = os.path.basename(bin_path).split('.')[0]
    save_dir = os.path.join(dest_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    file = open(os.path.join(dest_dir, '{}.txt'.format(dataset_name)), 'w')
    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
    for idx in tqdm(range(len(bins))):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, str(idx + 1).zfill(5) + '.jpg'), img)
        if idx % 2 == 0:
            label = 1 if issame_list[idx // 2] else -1
            file.write(str(idx + 1).zfill(5) + '.jpg' + ' ' + str(idx + 2).zfill(5) + '.jpg' + ' ' + str(label) + '\n')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.bin:
        load_image_from_bin(args.source, args.dest)
    else:
        load_mx_rec(args.source, args.dest)
