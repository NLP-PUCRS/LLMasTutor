CONFIG: dict = {
    'model_name': "",
    'num_labels': 0,
    'model_path': "",
    'state_dict_path': "",
    'seed': 0,
    'cluster_factor': 0.0,
    'predict_k': 'density_usnid',
    'output_path': "",
    'analyzed_path': "",
    'exp_name': "",
    'n_comp_pca_all': 0,
    'n_comp_pca_train_dev': 0,
    'embeddings_info': {
        "subset_all": {
            "mean": 0.0,
            'max': 0.0,
            '95th_percentile': 0.0,
            'qtd_truncated': 0.0,
            'embeddings_batch_save_mean_std': [],
            'cosine_stats': {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0
            },
            'cosine_hist': [],
            'cosine_bins': [],
            'vals_sample': [],
            'eigenvalues': [],
            'top_10_eigenvalues': [],
            'explained_variance_90': 0.0
        },
        "subset_train_dev": {
            "mean": 0.0,
            'max': 0.0,
            '95th_percentile': 0.0,
            'qtd_truncated': 0.0,
            'embeddings_batch_save_mean_std': [],
            'cosine_stats': {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0
            },
            'cosine_hist': [],
            'cosine_bins': [],
            'vals_sample': [],
            'eigenvalues': [],
            'top_10_eigenvalues': [],
            'explained_variance_90': 0.0
        },
        "subset_test": {
            "mean": 0.0,
            'max': 0.0,
            '95th_percentile': 0.0,
            'qtd_truncated': 0.0,
            'embeddings_batch_save_mean_std': [],
            'cosine_stats': {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0
            },
            'cosine_hist': [],
            'cosine_bins': [],
            'vals_sample': [],
            'eigenvalues': [],
            'top_10_eigenvalues': [],
            'explained_variance_90': 0.0
        },
    },
     
    'X_shape': None,
    'silhouette_global': 0.0,
    'inter_cluster_distance': 0.0,
    'intra_cluster_similarity_cos': 0.0,
    'cluster_keywords': {}
}