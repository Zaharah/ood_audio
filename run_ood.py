import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import numpy as np
# See installation guide for FAISS here: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
import faiss
import metrics

parser = argparse.ArgumentParser(description="OOD Detection for Audio")

parser.add_argument("--ind_dataset", 
    default="mswc_en", type=str,
    help="in-distribution dataset name")

parser.add_argument("--ood_dataset", 
    default="vocalsound", type=str,
    help="out-of-distribution dataset name")

parser.add_argument("--models_dir", 
    default="./models/", type=str,
    help="models directory path")

parser.add_argument("--features_dir", 
    default="./features/", type=str, 
    help="features directory path")

parser.add_argument("--k", 
    default=5, type=int, 
    help="number of nearest neighbors for ood")

args = parser.parse_args()

def run_deep_knn_ood(args):
    features_path = os.path.join(args.features_dir,
        f"{args.ind_dataset}_yamnet")
    
    tr_ind_feat = np.load(os.path.join(features_path, 
        "ind_train_features.npy"))
    ts_ind_feat = np.load(os.path.join(features_path, 
        "ind_test_features.npy"))
    ts_ood_feat = np.load(os.path.join(features_path, 
        f"{args.ood_dataset}_ood_test_features.npy")) 

    normalizer = lambda x: x / (np.linalg.norm(x, 
        ord=2, axis=-1, keepdims=True) + 1e-10)
    tr_ind_feat = normalizer(tr_ind_feat) 
    ts_ind_feat = normalizer(ts_ind_feat)
    ts_ood_feat = normalizer(ts_ood_feat)

    index = faiss.IndexFlatL2(tr_ind_feat.shape[1])
    index.add(tr_ind_feat)
    ind_D, _ = index.search(ts_ind_feat, args.k)
    ind_scores = -ind_D[:,-1]
    ood_D, _ = index.search(ts_ood_feat, args.k)
    ood_scores = -ood_D[:,-1]

    results = metrics.get_measures(
        ind_scores, ood_scores, 
        recall_level = 0.95)
    fpr95 = results["FPR"]
    auroc = results["AUROC"]
    print(f"FPR95: {fpr95} | AUROC: {auroc}")


if __name__ == "__main__":
    run_deep_knn_ood(args)