import numpy as np

from deslib.des.knora_e import KNORAE
from deslib.util.instance_hardness import kdn_score
from scipy.spatial.distance import squareform, pdist


class LORDS(KNORAE):
    """Local Overlap Reducing Dynamic Selection technique (LORDS).

        Similarly to KNORA-E, this method searches for a member of the ensemble
        that correctly classify all samples within the region of competence
        of the test sample. All classifiers that fulfill this condition are selected.
        If no classifier is able to label all neighbors correctly, the size of the
        region of competence is reduced by removing the neighbor with the highest
        instance hardness score, considered to be the least reliable neighbor, and the
        performance of the classifiers are re-evaluated. The outputs of the selected
        classifiers is combined using majority voting. If no classifier is selected,
        the whole pool is used for classification.

        Parameters
        ----------
         pool_classifiers : list of classifiers (Default = None)
            The generated_pool of classifiers trained for the corresponding
            classification problem. Each base classifiers should support the method
            "predict". If None, then the pool of classifiers is a bagging
            classifier.

        k : int (Default = 7)
            Number of neighbors used to estimate the competence of the base
            classifiers.

        IH : {'kdn', 'kdni', 'lsc', 'lsci'} (Default = 'kdn')
            Instance hardness measure, obtained using a leave-one-out evaluation
            over the training set, used to sort the instances within the local
            overlap reduction procedure.

            - 'kdn' will use the K-Disagreeing Neighbors (KDN) measure

            - 'kdni' will use the K-Disagreeing Neighbors-imbalance (KDNi) measure,
            adapted for class imbalanced problems

            - 'lsc' will use the Local Set Cardinality (LSC) measure

            - 'kdni' will use the Local Set Cardinality-imbalance (LSCi) measure,
            adapted for class imbalanced problems

        stats : Boolean (Default = False)
            Determines if statistics of the technique's performance will be saved.

        DFP : Boolean (Default = False)
            Determines if the dynamic frienemy pruning is applied.

        with_IH : Boolean (Default = False)
            Whether the region of competence's hardness level is used to
            decide between using the DS algorithm or the KNN for classification of
            a given query sample.

        safe_k : int (default = None)
            The size of the indecision region.

        IH_rate : float (default = 0.3)
            Region of competence's hardness threshold. If it is lower than the
            IH_rate the KNN classifier is used. Otherwise, the DS algorithm is
            used for classification.

        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        knn_classifier : {'knn', 'faiss', None} (Default = 'knn')
             The algorithm used to estimate the region of competence:

             - 'knn' will use :class:`KNeighborsClassifier` from sklearn
              :class:`KNNE` available on `deslib.utils.knne`

             - 'faiss' will use Facebook's Faiss similarity search through the
               class :class:`FaissKNNClassifier`

             - None, will use sklearn :class:`KNeighborsClassifier`.

        knne : bool (Default=False)
            Whether to use K-Nearest Neighbor Equality (KNNE) for the region
            of competence estimation.

        DSEL_perc : float (Default = 0.5)
            Percentage of the input data used to fit DSEL.
            Note: This parameter is only used if the pool of classifier is None or
            unfitted.

        n_jobs : int, default=-1
            The number of parallel jobs to run. None means 1 unless in
            a joblib.parallel_backend context. -1 means using all processors.
            Doesn’t affect fit method.

        References
        ----------

        Souza, Mariana A., et al. "Local overlap reduction procedure
        for dynamic ensemble selection." 2022 International Joint
        Conference on Neural Networks (IJCNN). IEEE, 2022.

        Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr.
        "From dynamic classifier selection to dynamic ensemble
        selection." Pattern Recognition 41.5 (2008): 1718-1731.

        Smith, Michael R., Tony Martinez, and Christophe Giraud-Carrier.
        "An instance level analysis of data complexity."
        Machine learning 95 (2014): 225-256.

        Leyva, Enrique, Antonio González, and Raul Perez. "A set of complexity
        measures designed for applying meta-learning to instance selection."
        IEEE Transactions on Knowledge and Data Engineering 27.2 (2014): 354-367.

        """

    def __init__(self, pool_classifiers=None, k=7, IH='kdn',
                 stats=False, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, random_state=None,
                 knn_classifier='knn', knne=False,
                 DSEL_perc=0.5, n_jobs=-1):

        super().__init__(pool_classifiers=pool_classifiers,
                         k=k,
                         DFP=DFP,
                         with_IH=with_IH,
                         safe_k=safe_k,
                         IH_rate=IH_rate,
                         random_state=random_state,
                         knn_classifier=knn_classifier,
                         knne=knne,
                         DSEL_perc=DSEL_perc,
                         n_jobs=n_jobs)
        self.IH = IH
        self.class_matrix_ = None
        self.class_sizes_ = None
        self.ih_ = None
        self.stats = stats

    def fit(self, X_dsel, y_dsel):

        # Fit DS technique
        super().fit(X_dsel, y_dsel)

        # Initialize stats
        if self.stats:
            self.stats_ = dict()

        # Obtain the instance hardness estimates in a leave-one-out
        # procedure over the training set
        
        self.class_matrix_ = np.zeros((self.n_samples_, self.n_classes_))
        self.class_matrix_[np.arange(self.n_samples_), self.DSEL_target_] = 1
        self.class_sizes_ = np.sum(self.class_matrix_, axis=0)

        loo_pdist = squareform(pdist(X_dsel))
        indices = np.argsort(loo_pdist, axis=1)

        loo_nn = indices[:, 1:]
        diff_class = np.tile(y_dsel, (loo_nn.shape[1], 1)).transpose() != y_dsel[loo_nn]

        e = 0.001
        f_adjust = lambda x: 1-(1 / (1+x))

        if self.IH.find('kdn') >= 0:
            k = 5
            diff_class = diff_class[:, :k]
            # Base KDN score
            kdn = np.sum(diff_class, axis=1) / k
            if self.IH.find('i') >= 0:
                kdn = kdn + e
                # Proportion of samples from the opposite class
                p_classes = self.class_sizes_[::-1]
                p_samples = p_classes[self.DSEL_target_]
                # KDNi score
                kdn_ = f_adjust(kdn * y_dsel.shape[0] / p_samples)
            else:
                # KDN score
                kdn_ = kdn
            self.ih_ = kdn_
        elif self.IH.find('lsc') >= 0:
            # Position of nearest enemy
            pos_ne = np.argmax(diff_class, axis=1).ravel() + 1
            # Base LSC score
            lsc = pos_ne/y_dsel.shape[0]
            if self.IH.find('i') >= 0:
                # Proportion of samples from the same class
                p_classes = self.class_sizes_
                p_samples = p_classes[self.DSEL_target_]
                # LSCi score
                lsc_ = (lsc * y_dsel.shape[0]) / p_samples

            else:
                # LSC score
                lsc_ = lsc
            self.ih_ = 1 - lsc_
        return self

    def _sort_roc(self, neighbors, distances=None):
        """Sorts the neighbors in the region of competence according
         to their estimated instance hardness, from lowest to highest.

        Parameters
        ----------

        neighbors : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors for each test sample,
            sorted by distance.

        distances : array of shape (n_samples, n_neighbors)
            Distances of the k nearest neighbors for each test sample,
            sorted by distance.

        Returns
        -------
        neighbors : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors for each test sample,
            sorted by instance hardness estimate

        distances : array of shape (n_samples, n_neighbors)
            Distances of the k nearest neighbors for each test sample.

        """

        ih_nn = self.ih_[neighbors]
        idx = np.argsort(ih_nn, axis=1)
        neighbors_ = neighbors[np.arange(len(idx)), idx.T].T
        if self.stats:
            self.stats_['neighborhood_hardness'] = ih_nn

        if distances is not None:
            distances_ = distances[np.arange(len(idx)), idx.T].T

        return neighbors_, distances_

    def estimate_competence(self, query, neighbors, distances=None,
                            predictions=None):
        """Competence estimation procedure of LORDS.
        Code adapted from the parent class method.
        The classifiers are considered competent if they correctly label all
        of the RoC. If none can be found, the hardest instance in the region
        is removed and the search is repeated in the new, smaller RoC.
        The implementation of this method first sorts the instances in the
        RoC in terms of their hardness estimates, from lowest to highest,
        and then singles out the classifier(s) that can correctly label the
        highest number of contiguous samples in the sorted neighborhood structure
        (or their competence level).

        Parameters
        ----------
        query : array of shape (n_samples, n_features)
                The test examples.

        neighbors : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors of each test sample

        distances : array of shape (n_samples, n_neighbors)
            Distances of the k nearest neighbors of each test
            sample.

        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifiers for all test examples.

        Returns
        -------
        competences : array of shape (n_samples, n_classifiers)
            Competence level estimated for each base classifier and test
            example.
        """

        # Re-order RoC according to the instances' hardness estimates
        neighbors_, distances_ = self._sort_roc(neighbors, distances)

        results_neighbors = self.DSEL_processed_[neighbors_, :]

        # Stores the stats from the technique
        if self.stats:
            self.stats_['sorted_neighbors'] = neighbors_
            self.stats_['neighbors'] = neighbors
            self.stats_['diff_order'] = np.any(neighbors_ != neighbors, axis=1)
            self.stats_['ratio_diff_order'] = np.sum(self.stats_['diff_order'])/(len(self.DSEL_target_)/4)
            self.stats_['distances'] = distances
            self.stats_['sorted_distances'] = distances_
            self.stats_['predictions'] = predictions

        # Similar procedure to the KNORA-E technique, but with the RoC ordered

        # Get the shape of the vector in order to know the number of samples,
        # base classifiers and neighbors considered.
        shape = results_neighbors.shape

        # add a row with zero for the case where the base classifier correctly
        # classifies the whole neighborhood. That way the search will always
        # find a zero after comparing to self.K + 1 and will return self.K
        # as the Competence level estimate (correctly classified the whole
        # neighborhood)
        addition = np.zeros((shape[0], shape[2]))
        results_neighbors = np.insert(results_neighbors, shape[1], addition,
                                      axis=1)

        # Look for the first occurrence of a zero in the processed predictions
        # (first misclassified sample). The np.argmax can be used here, since
        # in case of multiple occurrences of the maximum values, the indices_
        # corresponding to the first occurrence are returned.
        competences_ = np.argmax(results_neighbors == 0, axis=1)
        return competences_.astype(float)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from sklearn.linear_model import Perceptron
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import f1_score
    from deslib.util.aggregation import majority_voting
    from deslib.static.oracle import Oracle
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_moons

    def plot_dataset(X, y, ax=None, marker=None,
                     alpha=1.0, cmap='gnuplot', **kwargs):
        colors = cm.ScalarMappable(cmap=cmap).to_rgba(y)
        colors[:, 3] = alpha
        if ax is None:
            ax = plt.scatter(X[:, 0], X[:, 1], marker=marker,
                             c=colors, **kwargs)
        else:
            ax.scatter(X[:, 0], X[:, 1], marker=marker,
                       c=colors, **kwargs)

        return ax

    measures = ['kdn', 'kdni', 'lsc', 'lsci']
    size_classes = (1000, 100)
    test_size = 0.25
    ensemble_size = 100

    print('Size of classes: ', size_classes)
    print('Test size: ', test_size)
    print('Ensemble size: ', ensemble_size)

    X, y = make_moons(size_classes, shuffle=True,
                      noise=0.25, random_state=9)

    X_tra, X_tst, y_tra, y_tst = train_test_split(X, y,
                                                  test_size=test_size,
                                                  random_state=42)

    ensemble = BaggingClassifier(base_estimator=Perceptron(),
                                 n_estimators=ensemble_size,
                                 n_jobs=5,
                                 random_state=100)
    ensemble.fit(X_tra, y_tra)

    # Identify minority class samples
    mask_min_class = np.concatenate((y_tra, y_tst)) == 1

    # Calculate true KDN of the samples
    true_kdn, _ = kdn_score(np.concatenate((X_tra, X_tst)),
                            np.concatenate((y_tra, y_tst)), k=5)

    # Categorize minority class instances and imbalanced problem from:
    # Napierala, Krystyna, and Jerzy Stefanowski.
    # "Types of minority class examples and their influence on
    # learning classifiers from imbalanced data."
    # Journal of Intelligent Information Systems 46 (2016): 563-597.

    print('Dataset categorization')
    mask_safe = np.logical_and(mask_min_class,
                               true_kdn < 0.3)
    p_safe = np.sum(mask_safe)/np.sum(mask_min_class)
    print('% Safe ', 100*p_safe)
    tp = 'normal' if p_safe >= 0.6 else 'difficult'
    mask_borderline = np.logical_and(mask_min_class,
                                     np.logical_and(true_kdn >= 0.3,
                                                    true_kdn <= 0.6))
    print('% Borderline ', 100 * np.sum(mask_borderline) / np.sum(mask_min_class))
    mask_rare = np.logical_and(mask_min_class,
                               np.logical_and(true_kdn > 0.6,
                                              true_kdn < 1.0))
    print('% Rare ', 100 * np.sum(mask_rare) / np.sum(mask_min_class))
    mask_outlier = np.logical_and(mask_min_class,
                                  true_kdn == 1.0)
    print('% Outlier ', 100 * np.sum(mask_outlier) / np.sum(mask_min_class))
    print('Dataset category: ', tp)

    # Evaluate LORDS (with all four IH measures),
    # KNORA-E, Majority voting (MJ) and Oracle
    # using the F1 score
    techniques = dict()
    f1_all = dict()
    for m in measures:
        print(m)
        # Fit LORDS
        techniques[m] = LORDS(pool_classifiers=ensemble,
                              IH=m,
                              stats=False)
        techniques[m].fit(X_tra, y_tra)

        # Range of IH scores
        c = techniques[m].ih_
        print('Max ' + m + ': ' + str(np.max(c)))
        print('Min ' + m + ': ' + str(np.min(c)))

        # Plot data according to estimated IH measure
        markers = ['s', '^']
        fig, ax = plt.subplots(figsize=(5, 5))
        for i, marker in enumerate(markers):
            ax = plot_dataset(X_tra[y_tra == i, :], c[y_tra == i], ax=ax,
                              marker=marker, cmap='gnuplot')
        plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=np.min(c), vmax=np.max(c)),
                                       cmap='gnuplot'), ax=ax)
        plt.savefig(m + '-training-data.png', format='png')

        # Performance of LORDS
        y_out = techniques[m].predict(X_tst)
        f1_all['LORDS - ' + m] = f1_score(y_tst, y_out)

    # Performance of KNORA-E
    kne = KNORAE(pool_classifiers=ensemble)
    kne.fit(X_tra, y_tra)
    y_out = kne.predict(X_tst)
    f1_kne = f1_score(y_tst, y_out)
    f1_all['KNE'] = f1_kne

    # Performance of Majority Voting
    f1_mv = f1_score(y_tst, majority_voting(ensemble, X_tst))
    f1_all['MV'] = f1_mv

    # Performance of the Oracle
    oracle = Oracle(pool_classifiers=ensemble)
    oracle.fit(X_tra, y_tra)
    f1_oracle = f1_score(y_tst, oracle.predict(X_tst, y_tst))
    f1_all['Oracle'] = f1_oracle

    print('Overall F1 scores')
    print(f1_all)



