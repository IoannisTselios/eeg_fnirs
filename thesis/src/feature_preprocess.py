import os
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn import decomposition, preprocessing
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from statsmodels.stats.anova import AnovaRM


def quantile_bucket(data, buckets):
    """
    Returns the quantile bucketed data.
    """
    assert len(data) > buckets, "More buckets than data points, not sure how that is going to work"
    q = np.linspace(0.0, 1.0, buckets + 1)
    bins = np.quantile(data, q)
    bins = bins[1:-1] # Without this, the last bucket will contain only the max value, because of how np.digitize works (see bins' bounds), and the buckets' indexes will start at 1 instead of 0
    data = np.digitize(data, bins = bins)
    return data


# ---- Load ----

df = pd.read_csv(os.path.join("data", "processed", "eeg_features copy relative.csv"))
# df_fnirs = pd.read_csv(os.path.join("data", "processed", "fnirs_features copy.csv"))
# df_pupillometry = pd.read_csv(os.path.join("data", "processed", "pupillometry_features.csv"))

# ---- Merge ----

# df_eeg.id.astype('Int64')
# df_fnirs.id.astype('Int64')
# df_pupillometry.id.astype('Int64')
# df = pd.merge(left=df_eeg, right=df_pupillometry, left_on=['id', 'drug', 'time'], right_on=['id', 'drug', 'time'])
# df = pd.merge(left=df, right=df_fnirs, left_on=['id', 'drug', 'time'], right_on=['id', 'drug', 'time'])
del df_eeg, df_pupillometry, df_fnirs

# ---- [Data preparation] Substract baseline from corresponding post administration 1 and 2 ----

substract_baseline = True
use_median_baseline = False

if substract_baseline:
    
    # Save the average (with 'mean') or median (with 'median') baseline for the recordings that don't have a baseline # TODO : is that a good idea? Probably not
    median_baseline = df.drop(['drug', 'id'], axis=1).groupby("time").agg({'median'}).values[0]

    only_baseline_df = df[df['time'] == 0]
    only_post_one_df = df[df['time'] == 1]
    only_post_two_df = df[df['time'] == 2]
    del df

    for df in [only_post_one_df, only_post_two_df]:

        for index, row in df.iterrows():
            id = row['id']
            drug = row['drug']
            try:
                baseline_for_this_session = only_baseline_df.loc[(only_baseline_df['drug'] == drug) & (only_baseline_df['id'] == id)].values[0]
                df.loc[index, df.columns.values[3:]] -= baseline_for_this_session[3:]
            except Exception as e: # Most likely the baseline was not found
                # print(e)
                if use_median_baseline:
                    df.loc[index,df.columns.values[3:]] -= median_baseline
                else:
                    df.loc[index,df.columns.values[3:]] = [np.nan] * len(median_baseline)

    df = pd.concat([only_post_one_df, only_post_two_df])
    del only_baseline_df, only_post_one_df, only_post_two_df
        
df = df.dropna()
print("Number of data points: {}".format(len(df)))

# ---- [Data preparation] Normalize features ----

normalize = True

if normalize:
    ss = StandardScaler()
    for feature in df.columns:
        if feature in ['id', 'drug', 'time']:
            continue
        df[feature] = ss.fit_transform(df[feature].values.reshape(-1, 1))

df.to_csv(os.path.join("data", "processed", "lmm_data.csv"), index=False)
# df.to_excel(os.path.join("data", "processed", "lmm_data.xlsx"), index=False)

# ---- Visualizations ----

violin_plots = False

if violin_plots:
    ylabels = {'delta': 'Percentage of the total signal\'s power spectral density',
               'theta': 'Percentage of the total signal\'s power spectral density',
               'alpha': 'Percentage of the total signal\'s power spectral density',
               'ratio': 'Ratio',
               'pe': 'Permutation entropy',
               'se': 'Spectral entropy',
               'fnirs_1': 'Average slope of the hemoglobin signal',
               'pupillometry_score': 'Number of significant pupil dilations during the mental arithmetic tasks'}
    for feature in df.columns:
        if feature in ['id', 'drug', 'time']:
            continue

        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x="drug", y=feature, hue="time", palette='pastel')
        plt.title("Distribution of the values for the feature: {}".format(feature))
        plt.ylabel(ylabels[feature])
        plt.xlabel("Drug group")
        plt.savefig(r"H:\Dokumenter\Violin plots\Change since T0\\" + feature + ".png")

old_vizs = False

if old_vizs:
    drug_one_df = df[df['drug'] == 1]
    drug_two_df = df[df['drug'] == 2]
    drug_three_df = df[df['drug'] == 3]
    c = len(df.columns) - 3
    del df
    all_recordings_df = pd.concat([drug_one_df, drug_two_df, drug_three_df])

    time_dict = {0:0, 1:15, 2:60}
    color_dict = {0:"red", 1:"green", 2:"blue"}
    coord_dict = {"delta":[-0.26, 0.26],
                "theta":[-0.16,0.16],
                "alpha":[-0.08,0.08],
                "sigma":[-0.045,0.045],
                "beta":[-0.12,0.12],
                "se":[-0.18,0.18],
                "pe":[-0.07,0.07],
                "zc":[-0.13,0.13],
                "pupillometry_score":[-2.5,2.5],
                "fnirs_1":[-1.01, 1.01]
                }
    results = {0:{1:{},
                2:{}
                },
            1:{1:{},
                2:{}
                },
            2:{1:{},
                2:{}
                }
            }
    for feature in all_recordings_df.columns:
        for i in range(3):
            for j in range(1, 3):
                results[i][j][feature] = {"positive":0, "total":0}

    patient_level = False

    if patient_level:
        for patient_id in all_recordings_df['id'].unique():
            fig, axs = plt.subplots(3, c, constrained_layout=True)
            fig.suptitle("Patient number {}".format(patient_id))
            print("Patient number {}".format(patient_id))

            df = all_recordings_df[all_recordings_df['id'] == patient_id]

            for drug_id in range(3):

                for feature_id, feature in enumerate(all_recordings_df.columns):

                    if feature in ['id', 'drug', 'time']:
                        continue

                    y = []
                    x = []

                    for time in range(1, 3):

                        temp_df = df[df['time'] == time]
                        temp_df = temp_df[temp_df['drug'] == (drug_id + 1)]
                        temp_y = temp_df[feature].values
                        if len(temp_y) > 0:
                            results[drug_id][time][feature]["total"] += 1
                            results[drug_id][time][feature]["positive"] += int(temp_y[0] > 0)
                        del temp_df
                        y.append(np.mean(temp_y))
                        x.append(time_dict[time])

                    ax = axs[drug_id, feature_id - 3]
                    ax.scatter(x, y, c = color_dict[drug_id])
                    ax.axhline()
                    ax.set_ylim(coord_dict[feature])

                    if drug_id == 0:
                        ax.set_title("{}".format(feature))

                    ax.tick_params(bottom=False)          

                axs[drug_id, 0].set_ylabel("Drug {}".format(drug_id + 1))

            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()

    group_level = False

    if group_level:
        fig, axs = plt.subplots(3, c, constrained_layout=True)

        for drug_id, df in enumerate([drug_one_df, drug_two_df, drug_three_df]):

            for feature_id, feature in enumerate(df.columns):

                if feature in ['id', 'drug', 'time']:
                    continue

                y = []
                x = []
                weights = []

                for time in range(1, 3):

                    temp_df = df[df['time'] == time]
                    temp_y = temp_df[feature].values
                    # temp_y = quantile_bucket(temp_y, 10)
                    del temp_df
                    y.append(np.mean(temp_y))
                    x.append(time_dict[time])

                    # y = np.concatenate((y, temp_y), axis=None)
                    # x += [time_dict[time]] * len(temp_y)

                    weights.append(len(temp_y))

                    # y.append(temp_y)

                ax = axs[drug_id, feature_id - 3]

                # ax.boxplot(y)

                ax.scatter(x, y, c = color_dict[drug_id])
                # ax.axhline() # horizontal line at y = 0 by default
                ax.set_ylim(coord_dict[feature])

                # reg = LinearRegression().fit(np.array([x]).T, y, sample_weight=weights)
                # dummy_x = np.linspace(0, 60, 100)
                # ax.plot(dummy_x, reg.predict(np.array([dummy_x]).T), c = "grey")

                if drug_id == 0:
                    ax.set_title("{}".format(feature))

                ax.tick_params(bottom=False, labelbottom=False)#, left=False, labelleft=False)          

            axs[drug_id, 0].set_ylabel("Drug {} ({} recordings)".format(drug_id + 1, weights))
            
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

pca_viz = False

if pca_viz:
        
    plt.cla()
    pca = decomposition.PCA(n_components = 2, random_state = 1)
    data = np.column_stack([df[feature].values for feature in df.columns if feature not in ['id', 'drug', 'time']])
    X = pca.fit_transform(data)
    x = X[:, 0]
    y = X[:, 1]
    labels = df['drug'].values
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.scatter(x, y, marker="o", c=labels, cmap=ListedColormap(['red', 'blue', 'green']))
    plt.title("Data distribution ({}D reduced to {}D, {} points)".format(len(df.columns.values[3:]), 2, len(x)))
    plt.xlabel("PC 1\n\nFeatures are {}\n\nPC 1 is {}\n\nPC 2 is {}".format(df.columns.values[3:],
                                                                            [float(np.round(x, 1)) for x in pca.components_[0]],
                                                                            [float(np.round(x, 1)) for x in pca.components_[1]]))
    plt.ylabel("PC 2")
    plt.locator_params(nbins=3)
    plt.subplots_adjust(bottom=0.25)
    plt.grid()
    plt.show()

exit()

# ---- [Group-level analysis] ANOVA between drug groups for each feature ----

anova_per_feature = True

if anova_per_feature:

    all_recordings_df = deepcopy(df)

    # Correcting the unbalance across drug groups
    unbalanced_ids = [id for id in set(all_recordings_df['id'].values)
                        if (id not in all_recordings_df[all_recordings_df['drug'] == 1]['id'].values)
                        or (id not in all_recordings_df[all_recordings_df['drug'] == 2]['id'].values)
                        or (id not in all_recordings_df[all_recordings_df['drug'] == 3]['id'].values)]

    balanced_df = all_recordings_df[~all_recordings_df['id'].isin(unbalanced_ids)]

    # Correcting the unbalance across recording times
    unbalanced_ids = []
    for drug_id in [1, 2, 3]:
        temp_df = balanced_df[balanced_df['drug'] == drug_id]
        unbalanced_ids += [id for id in set(temp_df['id'].values)
                           if (id not in temp_df[temp_df['time'] == 1]['id'].values)
                           or (id not in temp_df[temp_df['time'] == 2]['id'].values)]

    df = balanced_df[~balanced_df['id'].isin(unbalanced_ids)]

    for feature in df.columns:

        if feature in ['id', 'drug', 'time']:
            continue

        temp_df = df.drop(columns=[c for c in df.columns if c not in ['id', 'drug', 'time', feature]])
        temp_df.drop_duplicates(subset=['id', 'drug', 'time'], inplace=True)
        
        # Standardize feature
        # scaler = StandardScaler()
        # X = np.atleast_2d(temp_df[feature].values).T
        # temp_df[feature] = scaler.fit_transform(X)

        print("\nSize of the dataset used for ANOVA RM by drug group, for feature {}".format(feature))
        print("Drug 1 : {}".format(temp_df[temp_df['drug'] == 1].shape[0]))
        print("Drug 2 : {}".format(temp_df[temp_df['drug'] == 2].shape[0]))
        print("Drug 3 : {}\n".format(temp_df[temp_df['drug'] == 3].shape[0]))

        # print(temp_df)

        print(AnovaRM(data=temp_df,
                    depvar=feature,
                    subject='id',
                    within=['drug', 'time']).fit())
        
        # fig, axs = plt.subplots(1, 2, constrained_layout=True)
        # fig.suptitle(feature)

        # for time in range(1, 3):

        #     y = []
        #     x = []

        #     for drug_id in range(1, 4):

        #         temp_df = all_recordings_df[all_recordings_df['time'] == time]
        #         temp_df = temp_df[temp_df['drug'] == drug_id]
        #         temp_y = temp_df[feature].values
        #         del temp_df
        #         y.append(temp_y)
        #         x.append("Drug {}".format(drug_id))

        #     print("\nANOVA test between drug groups based on feature {} :".format(feature))
        #     F, p = stats.f_oneway(y[0], y[1], y[2])
        #     print("F = {} ; p = {}".format(np.round(F, 3), np.round(p, 3)))


        #     ax = axs[time-1]
        #     ax.axhline()
        #     ax.boxplot(y)
        #     ax.set_xticklabels(x)
        #     ax.set_title("T{} - T0".format(time_dict[time]))

        # plt.show()

exit()

# ---- [Group-level analysis] ANOVA between drug groups with all features after PCA ----

anova_all_features = False

if anova_all_features:

    df = deepcopy(all_recordings_df)

    # Standardize features before PCA to account for different scales
    scaler = StandardScaler()
    X = df.drop(columns=['id', 'drug', 'time']).values
    X = scaler.fit_transform(X)

    # Apply PCA to reduce dimensionality to 1
    nb_pca_components = 1
    pca = decomposition.PCA(n_components = nb_pca_components, random_state = 1)
    X = pca.fit_transform(X)

    # Perform ANOVA test between groups
    reduced_df = pd.DataFrame({'drug':df['drug'].values, 'time':df['time'].values, 'value':X.flatten()})

    all_times = False

    if all_times:
        # Drug groups
        g1 = reduced_df[reduced_df['drug'] == 1]['value'].values
        g2 = reduced_df[reduced_df['drug'] == 2]['value'].values
        g3 = reduced_df[reduced_df['drug'] == 3]['value'].values

        plt.figure()
        plt.boxplot([g1, g2, g3])
        plt.xlabel("Drug group")
        plt.ylabel("First principal component from PCA")
        plt.grid()

        print("ANOVA test between drug groups (PCA-reduced features):")
        F, p = stats.f_oneway(g1, g2, g3)
        print(F, p)
    else:
        for time in [1, 2]:
            # Drug groups by time
            g1 = reduced_df[reduced_df['drug'] == 1][reduced_df['time'] == time]['value'].values
            g2 = reduced_df[reduced_df['drug'] == 2][reduced_df['time'] == time]['value'].values
            g3 = reduced_df[reduced_df['drug'] == 3][reduced_df['time'] == time]['value'].values

            plt.figure()
            plt.boxplot([g1, g2, g3])
            plt.xlabel("Drug group")
            plt.ylabel("First principal component from PCA")
            plt.grid()

            print("ANOVA test between drug groups (PCA-reduced features):")
            F, p = stats.f_oneway(g1, g2, g3)
            print(F, p)

    plt.show()

# ---- [Group-level analysis] Mixed linear model between drug groups for each recording time and feature ----

mlm = False # See R script instead

if mlm:

    for time_id in [1, 2]:

        df = all_recordings_df[all_recordings_df['time'] == time_id]

        for feature in df.columns:

            if feature in ['id', 'drug', 'time']:
                continue

            temp_df = df.drop(columns=[c for c in df.columns if c not in ['id', 'drug', feature]])

            md = smf.mixedlm("{} ~ drug".format(feature), temp_df, groups="drug")
            mdf = md.fit()
            print("\n\nFeature : {} - Time : {}\n\n".format(feature, time_id), mdf.summary(), "\n\n\n\n")

# ---- [Recording-level analysis] Define then count significant variations for each feature ----

variation_count = False
per_feature = False
all_features = False

if variation_count:

    df_variation_counts = deepcopy(all_recordings_df.drop(columns=['id']))

    for feature in df_variation_counts:
        if feature in ['drug', 'time']:
            continue

        threshold = np.std(all_recordings_df[feature].values) / 20 # TODO: Hyperparameter
        df_variation_counts[feature][df_variation_counts[feature] < -threshold] = -1
        df_variation_counts[feature][(df_variation_counts[feature] >= -threshold) & (df_variation_counts[feature] <= threshold)] = 0
        df_variation_counts[feature][df_variation_counts[feature] > threshold] = 1

    if per_feature:
        for drug_id in [1, 2, 3]:
            for time_id in [1, 2]:
                for feature in df_variation_counts:
                    if feature in ['drug', 'time']:
                        continue
                    print("\nVariation counts for drug {} and time {}".format(drug_id, time_dict[time_id]))
                    print(df_variation_counts[(df_variation_counts['drug'] == drug_id) & (df_variation_counts['time'] == time_id)][feature].value_counts())

    if all_features:
        print("\n")
        print(df_variation_counts.groupby("drug").value_counts())
        df_variation_counts.to_excel("test.xlsx")

# ---- Normalize data with quantile bucketing ---- TODO : DO NOT use if PCA is needed

temp = []
for feature in df.columns:
    if feature in ['id', 'drug', 'time']:
        continue
    feature_vector = df[feature].values

    feature_vector = quantile_bucket(feature_vector, 10) # TODO : hyperparameter
    temp.append(feature_vector)

X = np.column_stack(temp)
        
# ---- Normalize each feature in range [0, 1] ----

X = preprocessing.normalize(X, axis=1)

# ---- Dimensionality reduction with PCA ----

plt.cla()
nb_pca_components = 2 # TODO : hyperparameter
pca = decomposition.PCA(n_components = nb_pca_components, random_state = 1)
pca.fit(X)
X = pca.transform(X)
for i in range(nb_pca_components):
    print("PC {} in feature space: {}".format(i + 1, pca.components_[i]))
    print("Variance explained : {}".format(pca.explained_variance_[i]))

# ---- Visualize the PCA-transformed data ---

x = X[:, 0]
y = X[:, 1]
#z = X[:, 2]
labels = df['drug'].values
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()#(projection='3d')
ax.scatter(x, y, marker="o", c=labels, cmap=ListedColormap(['red', 'blue', 'green']))
plt.title("Data distribution ({}D reduced to {}D, {} points)".format(len(df.columns.values[3:]),
                                                          nb_pca_components,
                                                          len(x)
                                                          )
                                                          )
plt.xlabel("PC 1\n\nFeatures are {}\n\nPC 1 is {}    PC 2 is {}".format(df.columns.values[3:],
                                                                        [np.round(x, 1) for x in pca.components_[0]],
                                                                        [np.round(x, 1) for x in pca.components_[1]]
                                                                        )
                                                                        )
plt.ylabel("PC 2")
plt.locator_params(nbins=3)
plt.subplots_adjust(bottom=0.25)
plt.grid()
plt.show()

# ---- Save as a file ----

np.save(os.path.join("data", "processed", "X"), X)
np.save(os.path.join("data", "processed", "labels"), labels)