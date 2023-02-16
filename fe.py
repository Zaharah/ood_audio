import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from melfb import MelFilterbanks

parser = argparse.ArgumentParser(description="OOD Detection for Audio")

parser.add_argument("--ind_dataset", 
    type=str,
    default="mswc_en",
    help="in-distribution dataset name")

parser.add_argument("--ood_dataset", 
    type=str,
    default="fluent_speech",
    help="out-of-distribution dataset name")

parser.add_argument("--models_dir", 
    type=str,  default="./models/", 
    help="model directory path")

parser.add_argument("--features_dir", 
    type=str, default="./features/",  
    help="features directory path")

args = parser.parse_args()

def iterate_data_features(data_loader, model):
  feats = []
  for _, (x,_) in enumerate(tqdm(data_loader)):
    out = model(x, training=False)
    feats.append(out)
  return np.concatenate(feats)

def run(args):
    ind_train_dataset, ind_test_dataset, num_classes = ... # Load in-distribution audio dataset here as TFDS objects
    ood_test_dataset = ... # Load out-of-distribution audio dataset here as TFDS objects

    model = tf.keras.models.load_model(os.path.join(args.models_dir, 
        f"{args.ind_dataset}_yamnet.h5"), 
        custom_objects={"MelFilterbanks":MelFilterbanks})
    model = tf.keras.models.Model(inputs=model.input, 
        outputs=model.get_layer("global_average_pooling2d").output) 
    model.summary()

    features_path = os.path.join(args.features_dir, 
        f"{args.ind_dataset}_yamnet")
    if not os.path.exists(features_path):
        os.makedirs(features_path)

    if not os.path.isfile(os.path.join(features_path, "ind_train_features.npy")):
        print("ID training set features...")
        tr_ind_feat = iterate_data_features(ind_train_dataset, model)
        np.save(os.path.join(features_path, "ind_train_features"), tr_ind_feat)
    if not os.path.isfile(os.path.join(features_path, "ind_test_features.npy")):
        print("ID testing set features...")
        ts_ind_feat = iterate_data_features(ind_test_dataset, model)
        np.save(os.path.join(features_path, "ind_test_features"), ts_ind_feat)

    print("OOD testing set features...")
    ts_ood_feat = iterate_data_features(ood_test_dataset, model)
    np.save(os.path.join(features_path,
        f"{args.ood_dataset}_ood_test_features"), ts_ood_feat)

if __name__ == "__main__":
    run(args)