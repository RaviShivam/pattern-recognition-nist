from prutils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


kpca_full_raw = run_KPCA_experiment(LDA(), RAW_PIXELS_DATASET, max_components = 50)
kpca_batch_raw = run_KPCA_experiment(LDA(), RAW_PIXELS_DATASET, max_components = 50)

kpca_full_features = run_KPCA_experiment(LDA(), IM_FEATURES_DATASET, max_components = 50)
kpca_batch_features = run_KPCA_experiment(LDA(), IM_FEATURES_DATASET, max_components = 50)

plot_performance([kpca_full_raw, kpca_batch_raw, kpca_full_features, kpca_batch_features],
                 show_results=False, save_to_file="lda_kpca")
