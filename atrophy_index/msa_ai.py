import numpy as np
import statsmodels.formula.api as smf
import pandas as pd


# ---------------------------------------------------------------------------
# Default feature set.
#
# Each entry maps a composite feature name to a list of spreadsheet columns.
# Columns within a group are summed together (see __array_prep__), so e.g.
# "feature_1" becomes a single volume = Pallidum + Putamen.
#
# Anatomically these capture the three canonical MSA degeneration targets:
#   feature_1 -> striatopallidal (putaminal / pallidal) atrophy
#   feature_2 -> cerebellar atrophy
#   feature_3 -> brainstem atrophy
#
# Column names must match the reference/scoring spreadsheet headers exactly,
# including the trailing underscores.
# ---------------------------------------------------------------------------
base_set = {"feature_1": ["PallidumTotalVolume_", "PutamenTotalVolume_"],
            "feature_2": ["CerebellumTotalVolume_"],
            "feature_3": ["BrainstemVolume_"]}


# MSA atrophy index class
class msai_ai_linear():
    """
    MSA Atrophy Index (MSA-AI) estimator based on normative (w-score) modeling.

    For each composite feature, an OLS model is fit on a reference population
    that predicts expected regional volume from demographics. For a new
    subject, the index is the weighted AVERAGE of the w-scores across features:

        wscore_i = (observed_i - predicted_i) / SD_reference_i
        MSA-AI   = sum_i ( weight_i * wscore_i ),  with sum_i(weight_i) = 1

    Weights are normalized internally to sum to 1, so equal weights give a
    plain average and the index stays on the same scale as a single regional
    w-score (i.e. interpretable in SD-like units).

    Sign convention: atrophied regions have observed < predicted, so the
    w-score is negative. More negative MSA-AI => more atrophy.
    """

    def __init__(self, reference_sheet,
                 feature_set=base_set,
                 a_fun=np.sum,
                 formula="{} ~ 1 + Age",
                 age_lim=[40, 80],
                 weight=[]):

        # Load the reference (normative) cohort and keep only subjects within
        # the age range over which the normative models are considered valid.
        self.reference_data = pd.read_excel(reference_sheet)
        self.reference_data = self.reference_data[
            (self.reference_data.Age >= age_lim[0]) &
            (self.reference_data.Age <= age_lim[1])
        ]

        self.feature_set = feature_set

        # Build the reference design matrix and capture the per-feature
        # standard deviations used to normalize the w-scores later.
        X_ref, Z_std = self.__array_prep__(dataframe=self.reference_data,
                                           is_training=True)
        self.Z_std = Z_std

        # Assemble a labeled dataframe for statsmodels: one column per feature,
        # plus the Age and Sex covariates appended during training prep.
        columns = list(self.feature_set.keys())
        columns.append("Age")
        columns.append("Sex")
        data_ref = pd.DataFrame(data=X_ref, columns=columns)

        # Fit one normative model per feature.
        #
        # NOTE on formula: the string is templated with each feature name, so
        # the default "{} ~ 1 + Age" fits volume against an intercept and age.
        # To additionally adjust for sex, pass formula="{} ~ 1 + Age + Sex".
        # The Sex covariate is always prepared, but only enters a model if the
        # formula references it.
        self.__model_fit__(data_ref, formula)

        # NOTE on a_fun: the function used to aggregate the weighted w-scores
        # across features (applied along axis=1). The default np.sum is the
        # intended choice: because the weights are normalized to sum to 1
        # (see below), summing the weighted w-scores yields a weighted AVERAGE.
        # Override only if you want different aggregation behavior, and be aware
        # it combines with the already-normalized weights. For example np.mean
        # would divide by the number of features a second time, shrinking the
        # index; np.max/np.min would instead report the single most/least
        # atrophied weighted region rather than an average.
        self.a_fun = a_fun

        # NOTE on weight: one weight per feature, giving each region's relative
        # contribution to the index. The list length must equal the number of
        # features. The default is equal weighting (all 1s). Raw weights are
        # normalized to sum to 1 below, so the index is a weighted AVERAGE of
        # the w-scores: only the ratios between weights matter, not their scale
        # (e.g. [2, 1, 1] and [0.5, 0.25, 0.25] give the same result).
        n_features = len(self.feature_set.keys())
        if len(weight) == 0:
            weight = [1 for _ in range(n_features)]
        if not (len(weight) == n_features):
            raise Exception("Weights must have same length as feature_set")
        weight = np.asarray(weight, dtype=float)
        self.weight = weight / weight.sum()

    def __array_prep__(self, dataframe, is_training=False):
        """
        Build the feature matrix from the requested columns.

        For each composite feature, the member columns are summed into a single
        volume vector. During training the Age and Sex covariates are appended
        as extra columns so the fitting dataframe holds everything statsmodels
        needs.

        Returns
        -------
        X_ref : ndarray, shape (n_subjects, n_features [+2 if training])
        Z_std : ndarray, shape (n_features,)
            Per-feature standard deviation of the summed composite volumes.
        """
        X_ref = []
        for d_key in self.feature_set.keys():
            key_set = self.feature_set[d_key]
            # Accumulate the member columns of this feature into one vector.
            vol_vec = np.zeros([len(dataframe), 1])
            for s_key in key_set:
                if s_key not in dataframe.keys():
                    raise Exception(
                        "{} - {} is not matching column header".format(d_key, s_key)
                    )
                vol_vec = vol_vec + dataframe[s_key].to_numpy().reshape(-1, 1)
            X_ref.append(vol_vec)

        # Stack the composite volumes into a (subjects x features) matrix.
        X_ref = np.concatenate(X_ref, axis=1)

        # Standard deviation per feature, taken on the composite volumes and
        # before any covariate columns are appended.
        Z_std = X_ref.std(axis=0)

        if is_training:
            # Append Age and a binary Sex indicator (True/1 for "M").
            X_ref = np.concatenate(
                (X_ref,
                 dataframe.Age.to_numpy().reshape(-1, 1),
                 (dataframe.Sex.to_numpy().reshape(-1, 1) == "M")),
                axis=1
            )
        return X_ref, Z_std

    def __model_fit__(self, data, formula):
        """
        Fit one OLS normative model per feature, templating the feature name
        into the formula string.
        """
        features = self.feature_set.keys()
        self._mdls_ = []
        for feat in features:
            print("Fitting model for formula : {}".format(formula.format(feat)))
            mdl = smf.ols(formula=formula.format(feat), data=data).fit()
            self._mdls_.append(mdl)

    def compute_msa_ai(self, data):
        """
        Compute the MSA-AI for one or more subjects.

        Returns the aggregated index (one value per subject). The per-feature
        w-score matrix is also stored on self.last_wscores_ so individual
        regional deviations can be retrieved after a call.

        More negative output => more atrophy.
        """
        # Covariate grid used for prediction. The Sex column uses the same
        # binary "M" indicator as training so the design matches.
        age_exog = data.Age.to_numpy()
        sex_exog = (data.Sex.to_numpy() == "M").astype(np.float32)
        grid = pd.DataFrame({"Age": age_exog, "Sex": sex_exog})

        # Observed composite volumes for the subjects being scored.
        X, S = self.__array_prep__(data)

        V = []
        for i, mdl in enumerate(self._mdls_):
            # w-score: (observed - normative prediction) / reference SD.
            wscore = (X[:, i] - mdl.predict(grid).to_numpy()) / self.Z_std[i]
            V.append(wscore.reshape(-1, 1))
        # Per-feature w-scores, shape (subjects x features). Kept for later use.
        V = np.concatenate(V, axis=1)
        self.last_wscores_ = V

        # Broadcast the per-feature weights across all subjects.
        W = np.tile(self.weight, (V.shape[0], 1))

        # Aggregate the weighted w-scores across features into the index.
        return self.a_fun(V * W, axis=1)

    def score_to_spreadsheet(self, data, output_path, subject_col="Subject"):
        """
        Score subjects and write a results spreadsheet with the columns:
        Subject, Age, Sex, wscore_pallidum_putamen, wscore_cerebellum,
        wscore_brainstem, MSA_AI.

        Parameters
        ----------
        data : DataFrame
            Subjects to score (same volume/Age/Sex columns as the reference).
        output_path : str
            Destination .xlsx path.
        subject_col : str
            Name of the subject identifier column in `data`. If absent, a
            sequential index is used instead.
        """
        # Compute the index; this also populates self.last_wscores_.
        msa_ai = self.compute_msa_ai(data)
        wscores = self.last_wscores_

        # Use the subject id column if present, otherwise number the rows.
        if subject_col in data.columns:
            subjects = data[subject_col].to_numpy()
        else:
            subjects = np.arange(1, len(data) + 1)

        out = pd.DataFrame({
            "Subject": subjects,
            "Age": data.Age.to_numpy(),
            "Sex": data.Sex.to_numpy(),
            "wscore_pallidum_putamen": wscores[:, 0],
            "wscore_cerebellum": wscores[:, 1],
            "wscore_brainstem": wscores[:, 2],
            "MSA_AI": msa_ai,
        })

        out.to_excel(output_path, index=False)
        print("Saved scored spreadsheet to: {}".format(output_path))
        return out