import datetime
import pickle
import pandas as pd


class Water_Asset_Data:
    def __init__(self, train_feature_file, train_label_file, test_feature_file, cat_cols, ord_cols, bool_cols, num_cols, label_col, id_col):
        '''create training and testing panda dataframes'''
        # Save data features by type and label
        print("Water_Asset_Data: __init__ ...")

        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.bool_cols = list(bool_cols)
        self.ord_cols = list(ord_cols)
        self.label_col = label_col
        self.id_col = id_col
        self.feature_cols = cat_cols + num_cols + bool_cols
        self.label_names = []

        self.saved_train_file_name = 'clean_wp_train_features_df_'
        self.saved_test_file_name = 'clean_wp_test_features_df_'
        self.saved_object_file_name = 'clean_wp_data_object_'

        # Approximate GPS bounderies of Tanzania (hardcoded)
        self.gps_bounderies_dict = {
            'lat_min': -12, 'lat_max': 0, 'lon_min': 30, 'lon_max': 40, 'height_min': 0}

        # Save our train and test data frames
        self.train_feature_df = self._create_train_feature_df(
            train_feature_file)
        self.label_df = self._create_label_df(train_label_file)
        self.test_feature_df = self._create_test_feature_df(test_feature_file)

    def _create_train_feature_df(self, train_feature_file, clean_features=True, shuffle_data=True):
        '''creates training feature dataframe and cleans it from missing values'''
        print("Water_Asset_Data: _create_train_feature_df ...")

        train_feature_df = self._load_data_file(train_feature_file)
        # train_df = self._merge_df(
        #    train_feature_df, train_target_df, self.id_col)
        if clean_features:
            train_feature_df = self._clean_data(train_feature_df)
        if shuffle_data:
            train_feature_df = self._shuffle_data(train_feature_df)
        return train_feature_df

    def _create_label_df(self, train_target_file):
        '''creates the label dataframe'''
        print("Water_Asset_Data: _create_label_df ...")
        train_target_df = self._load_data_file(train_target_file)
        self.label_names = train_target_df[self.label_col].unique()
        return train_target_df

    def _create_test_feature_df(self, test_feature_file, clean_features=True):
        '''creates the test dataframes and cleans it from missing values'''
        print("Water_Asset_Data: _create_test_feature_df ...")

        test_df = self._load_data_file(test_feature_file)
        if clean_features:
            self._clean_data(test_df)
        return test_df

    def _select_features(self, n_best_features=20):
        '''Select n most important features based on a basic random forest model'''
        print("Water_Asset_Data: _select_features ...")

        target_df = self.train_feature_df[self.label_col]
        feature_df = self.train_feature_df.drop(
            [self.id_col, self.label_col], axis=1)

        # Simple split of the training set
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df, target_df, shuffle=False)

        # Select best encoded features using RFE
        estimator = RandomForestClassifier(n_jobs=-1)
        selector = RFE(estimator, n_best_features, step=1)
        selector = selector.fit(X_train, y_train)

        # Drop least important features
        cols_to_drop = list(compress(list(feature_df.columns), ~(
            selector.get_support(indices=False))))
        self.train_feature_df.drop(cols_to_drop, inplace=True, axis=1)
        self.test_feature_df.drop(cols_to_drop, inplace=True, axis=1)

    def _encode_features(self):
        '''Preprocess and encode features'''

        train_len = self.train_feature_df.shape[0]

        # Encode label feature
        print("Water_Asset_Data: _encode_features label {}".format(self.label_col))
        self.label_df[self.label_col] = pd.Categorical(
            self.label_df[self.label_col])  # .codes
        #encoded_train_feature_df[self.label_col] = self.train_feature_df[self.label_col]

        # concatenate train and test data frames
        merged_df = pd.concat(
            [self.train_feature_df, self.test_feature_df], axis=0)

        # Encode boolean features
        print("Water_Asset_Data: _encode_features boolean {}".format(self.bool_cols))
        for col in self.bool_cols:
            merged_df[col] = merged_df[col] * 1

        #  numeric features
        print("Water_Asset_Data: _encode_features numeric {}".format(self.num_cols))
        merged_df[self.num_cols] = merged_df[self.num_cols].apply(
            pd.to_numeric)

        #  ordinal features
        print("Water_Asset_Data: _encode_features ordinal {}".format(self.ord_cols))
        merged_df[self.ord_cols] = merged_df[self.ord_cols].apply(
            pd.to_numeric)

        # One hot encode all other categorical features
        print("Water_Asset_Data: _encode_features categorical {}".format(self.cat_cols))
        encoded_cat_feature_df = pd.get_dummies(merged_df[self.cat_cols])
        merged_df.drop(self.cat_cols, axis=1, inplace=True)

        encoded_df = pd.concat([merged_df, encoded_cat_feature_df], axis=1)

        # Save encoded dfs
        self.train_feature_df = encoded_df.iloc[:train_len, :]
        self.train_feature_df = self._merge_df(
            train_feature_df=self.train_feature_df, train_label_df=self.label_df, merge_on_feature=self.id_col)
        self.test_feature_df = encoded_df.iloc[train_len:, :]

        print("Water_Asset_Data: _encode_features train set shape {} , test set shape {}".format(
            self.train_feature_df.shape, self.test_feature_df.shape))

    def _engineer_features(self):
        '''Generate features for test and training dataframes'''
        print("Water_Asset_Data: _engineer_features ...")
        self.train_feature_df = self._engineer_age_feature(
            self.train_feature_df)
        self.test_feature_df = self._engineer_age_feature(self.test_feature_df)

    def _load_data_file(self, file_name, verbose=True):
        '''Load Pickle data file'''
        file_df = pd.read_pickle(file_name)
        if verbose:
            print("Water_Asset_Data: _load_data_file - pickle file {} loaded with shape {}".format(file_name, file_df.shape))
        return file_df

    def _save_dfs_to_file(self, clean_dir, verbose=False):
        '''Save dataframes to pickle data file'''
        print("Water_Asset_Data: _save_df_to_file to {}".format(clean_dir))

        file_time = datetime.datetime.now().strftime("%y%m%d%H")
        train_pkl_file = clean_dir + self.saved_train_file_name + file_time + '.pkl'
        test_pkl_file = clean_dir + self.saved_test_file_name + file_time + '.pkl'
        # Save cleaned panda dfs in repository
        try:
            self.train_feature_df.to_pickle(train_pkl_file)
            self.test_feature_df.to_pickle(test_pkl_file)
            if verbose:
                print(
                    "Water_Asset_Data: _save_df_to_file - {} and {} saved".format(train_pkl_file, test_pkl_file))
        except:
            print("Water_Asset_Data: _save_df_to_file {} failed".format(
                train_pkl_file, test_pkl_file))

    def _save_object_to_file(self, clean_dir, verbose=False):
        # Save the wp_data object in the clean-data repository
        print("Water_Asset_Data: _save_object_to_file to {}".format(clean_dir))

        file_time = datetime.datetime.now().strftime("%y%m%d%H")
        object_pkl_file = clean_dir + self.saved_object_file_name + file_time + '.pkl'
        # Save cleaned Water_Asset_Data object in repository
        with open(object_pkl_file, 'wb') as f:
            pickle.dump(wp_data, f, pickle.HIGHEST_PROTOCOL)

    def _merge_df(self, train_feature_df, train_label_df, merge_on_feature):
        '''Merge the training and label df on a specific key'''
        merged_df = pd.merge(
            left=train_feature_df, right=train_label_df, how='inner', on=merge_on_feature)
        return merged_df

    def _shuffle_data(self, df):
        '''Shuffle the observation in the dataframe'''
        return shuffle(df)

    def _clean_data(self, df):
        '''Impute missing values'''
        print("Water_Asset_Data: _clean_data ...")
        col_names = df.columns.tolist()

        # Transform ordinal values as object values
        for col in self.ord_cols:
            if col in col_names:
                df[col] = df[col].astype('str')

        # Normalise string categorical features to lower case
        for col in cat_cols:
            if col in col_names:
                df[col] = df[col].str.lower()

        # Transform all numerical values into numeric values
        df[self.num_cols] = df[self.num_cols].apply(pd.to_numeric)

        # Remove scheme_name feature (too many missing values)
        self._drop_features(df, ['scheme_name'])

        # replace invalid construction_years (0) with the median construction year
        valid_construction_year_df = df[df['construction_year'] > 0]
        median_construction_year_value = int(
            valid_construction_year_df['construction_year'].median())
        df['construction_year'].replace(
            0, median_construction_year_value, inplace=True)

        # replace likely invalid GPS values (out of Tanzania boundaries)
        self._clean_GPS_features(df, verbose=_debug)

        # replace funder feature missing values with its highest occurrence
        self._impute_null_with_highest_frequency_value(
            df, 'funder', verbose=_debug)
        # replace installer feature missing values with its highest occurrence
        self._impute_null_with_highest_frequency_value(
            df, 'installer', verbose=_debug)
        # replace public_meeting feature missing values by highest occurrence
        self._impute_null_with_highest_frequency_value(
            df, 'public_meeting', verbose=_debug)
        # replace scheme_management feature missing values by highest occurrence for each funder category
        self._impute_null_with_highest_frequency_value(
            df, 'scheme_management', 'funder', verbose=_debug)
        # replace permit feature missing values with highest occurrence for each installer category
        self._impute_null_with_highest_frequency_value(
            df, 'permit', 'installer', verbose=_debug)
        # replace subvillage feature missing values with highest occurrence for each region category
        self._impute_null_with_highest_frequency_value(
            df, 'subvillage', 'region', verbose=_debug)

        # Check if the training set has any missing values
        num_null_val = df.isnull().sum().sort_values(ascending=False)
        if _debug:
            display(num_null_val.head())
        assert (sum(num_null_val) ==
                0), "Water_Asset_Data: _clean_data - dataframe has missing values"

        print("Water_Asset_Data: _clean_data - data frame shape {}".format(df.shape))
        print("Water_Asset_Data: _clean_data - data frame feature list {} \n".format(df.columns.tolist()))
        return(df)

    def _engineer_age_feature(self, df):
        '''Generate new age features'''
        print(
            "Water_Asset_Data: _engineer_age_feature \'pump_age\' and \'year_recorded\' ...")

        # Extract the of date_recorded feature and create a new feature: year_recorded
        df['year_recorded'] = df['date_recorded'].apply(self._extract_year)
        if 'year_recorded' not in self.num_cols:
            self.num_cols += ['year_recorded']

        # Compute the age of the pumps (in years) based on the construction_year and year of the obervation extracted
        df['pump_age'] = abs(df['year_recorded'] - df['construction_year'])
        if 'pump_age' not in self.num_cols:
            self.num_cols += ['pump_age']

        return(df)

    def _shrink_cat_features(self, cat_feature_list, threshold):
        '''Replace categorical feature values with number of occurrence below the treshhold provided '''
        print("_shrink_cat_features: _shrink_cat_features {}".format(cat_feature_list))
        for feature in cat_feature_list:
            self.train_feature_df[feature] = self._shrink_cat_feature(
                self.train_feature_df, feature, threshold)
            self.test_feature_df[feature] = self._shrink_cat_feature(
                self.test_feature_df, feature, threshold)

    def _shrink_cat_feature(self, df, cat_feature, threshold):
        '''Replace categorical feature values with number of occurrence below the treshhold provided '''
        print("Water_Asset_Data: shrink_cat_feature {}".format(cat_feature))
        s = (df[cat_feature].value_counts() <= threshold)
        list_to_replace = list(s[s == True].index)
        shrunk_cat_feature = df[cat_feature].apply(lambda x: (
            cat_feature + "_rare") if x in list_to_replace else x)
        return(shrunk_cat_feature)

    def _drop_features(self, df, feature_list):
        '''Remove feature list from data frames'''
        print("Water_Asset_Data: _drop_features ...")
        df.drop(feature_list, axis=1, inplace=True)
        for feature in feature_list:
            if feature in self.cat_cols:
                self.cat_cols.remove(feature)
            if feature in self.num_cols:
                self.num_cols.remove(feature)
            if feature in self.bool_cols:
                self.bool_cols.remove(feature)
            if feature in self.ord_cols:
                self.ord_cols.remove(feature)
        print("Water_Asset_Data: _drop_features - {} were dropped".format(feature_list))

    def _clean_GPS_features(self, df, verbose=False):
        '''Generate GPS features which seem out of Tanzania boundaries'''
        correct_gps_df = df[
            (df['latitude'] > self.gps_bounderies_dict['lat_min']) & (df['latitude'] < self.gps_bounderies_dict['lat_max']) &
            (df['longitude'] > self.gps_bounderies_dict['lon_min']) & (df['longitude'] < self.gps_bounderies_dict['lon_max']) &
            (df['gps_height'] > self.gps_bounderies_dict['height_min'])]

        # mean of gps coordinate types in each Tanzanian basin
        mean_correct_gps_df = correct_gps_df.groupby(
            ['basin'])['latitude', 'longitude', 'gps_height'].mean()

        # Replace likely invalid GPS values for each basin by basin GPS means
        basin_list = df['basin'].unique()
        for i in basin_list:
            correct_lon = mean_correct_gps_df.loc[i, 'longitude']
            if verbose:
                print("Updating invalid longitutes for basin {} by mean {}".format(
                    i, correct_lon))
            df[((df['longitude'] < self.gps_bounderies_dict['lon_min']) | (df['longitude'] > self.gps_bounderies_dict['lon_max'])) &
                (df['basin'] == i)]['longitude'] = correct_lon

            correct_lat = mean_correct_gps_df.loc[i, 'latitude']
            if verbose:
                print("Updating invalid latitudes for basin {} by mean {}".format(
                    i, correct_lat))
            df[((df['latitude'] < self.gps_bounderies_dict['lat_min']) | (df['latitude'] > self.gps_bounderies_dict['lat_max'])) &
                (df['basin'] == i)]['latitude'] = correct_lat

            correct_height = mean_correct_gps_df.loc[i, 'gps_height']
            if verbose:
                print("Updating invalid heights for basin {} by mean {}".format(
                    i, correct_height))
            df[(df['gps_height'] > self.gps_bounderies_dict['height_min']) &
               (df['basin'] == i)]['gps_height'] = correct_height

        return(df)

    def _extract_year(self, date):
        return (int(date.split('-')[0]))

    def _remove_list(self, orginal_list, list_to_be_removed):
        '''Remove a sublist from a list'''
        for i in list_to_be_removed:
            orginal_list.remove(i)

    def _impute_null_with_highest_frequency_value(self, df, impute_feature, frequency_feature=[], verbose=False):
        '''impute missing values based on the hightest occurrence of this feature in a seperate data slice'''
        impute_cat_set = df[impute_feature].nunique()

        if frequency_feature:
            feat_cat_list = df[frequency_feature].unique()
            for i in feat_cat_list:
                feat_cat = df[df[frequency_feature] == i][impute_feature]
                highest_feat_cat_occurrence = df[frequency_feature].value_counts(
                ).idxmax()
                if sum(feat_cat.notnull()):
                    highest_feat_cat_occurrence = feat_cat.value_counts().idxmax()
                if verbose:
                    print("Replacing {} feature null values: {} imputed by {}".format(
                        impute_feature, i, highest_feat_cat_occurrence))
                df[impute_feature].fillna(
                    value=highest_feat_cat_occurrence, inplace=True)
        else:
            highest_feat_cat_occurrence = df[impute_feature].value_counts(
            ).idxmax()
            if verbose:
                print("Replacing {} feature null values by {}".format(
                    impute_feature, highest_feat_cat_occurrence))
            df[impute_feature].fillna(
                value=highest_feat_cat_occurrence, inplace=True)

        # Sanity check: null values have been replaced
        assert impute_cat_set == df[impute_feature].nunique(
        ), "Water_Asset_Data: - impute_null_with_highest_frequency_value: null values replaced assert failed"
