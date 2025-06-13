import numpy as np
import statsmodels.formula.api as smf
import pandas as pd


base_set = {"feature_1": ["PallidumTotalVolume_", "PutamenTotalVolume_"],
            "feature_2": ["CerebellumTotalVolume_"],
            "feature_3": ["BrainstemVolume_"]}


class msai_ai_linear():

    def __init__(self, reference_sheet,
                 feature_set=base_set,
                 a_fun=np.sum,
                 formula="{} ~ 1 + Age",
                 age_lim=[40,80]):

        self.reference_data = pd.read_excel(reference_sheet)
        self.reference_data = self.reference_data[(self.reference_data.Age>=age_lim[0]) & (self.reference_data.Age<=age_lim[1])]

        self.feature_set = feature_set

        X_ref, Z_std = self.__array_prep__(dataframe=self.reference_data, is_training=True)
        self.Z_std = Z_std

        columns = list(self.feature_set.keys())
        columns.append("Age")
        columns.append("Sex")
        data_ref   = pd.DataFrame(data=X_ref, columns=columns)

        print(data_ref)

        self.__model_fit__(data_ref, formula)
        self.a_fun = a_fun


    def __array_prep__(self, dataframe, is_training=False):
        X_ref = []
        for i, d_key in enumerate(self.feature_set.keys()):
            key_set = self.feature_set[d_key]
            vol_vec = np.zeros([len(dataframe), 1])
            for j, s_key in enumerate(key_set):
                if not s_key in dataframe.keys():
                    raise Exception("{} - {} is not matching column header".format(d_key, s_key))
                vol_vec = vol_vec + dataframe[s_key].to_numpy().reshape(-1, 1)
            X_ref.append(vol_vec)

        X_ref = np.concatenate(X_ref, axis=1)
        Z_std = X_ref.std(axis=0)
        if is_training:
            X_ref = np.concatenate((X_ref,
                                    dataframe.Age.to_numpy().reshape(-1, 1),
                                    dataframe.Sex.to_numpy().reshape(-1,1)=="M"), axis=1)
        return X_ref, Z_std


    def __model_fit__(self, data, formula):
        features = self.feature_set.keys()
        self._mdls_ = []
        for i, feat in enumerate(features):
            print("Fitting model for formula : {}".format(formula.format(feat)))
            mdl = smf.ols(formula=formula.format(feat), data=data).fit()
            print(mdl.summary())
            self._mdls_.append(mdl)



    def compute_msa_ai(self, data, weight=[0.3, 0.3, 0.3]):
            age_exog = data.Age.to_numpy()
            sex_exog = (data.Sex.to_numpy()=="M").astype(np.float32)
            grid = pd.DataFrame({"Age": age_exog, "Sex": sex_exog})
            X, S = self.__array_prep__(data)

            V = []
            for i, mdl in enumerate(self._mdls_):
                wscore = (X[:,i] - mdl.predict(grid).to_numpy()) / self.Z_std[i]
                V.append(wscore.reshape(-1,1))
            V = np.concatenate(V,axis=1)
            W = np.tile(weight, (V.shape[0], 1))

            return self.a_fun(V * W,axis=1)

