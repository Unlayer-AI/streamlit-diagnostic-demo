import shap
import shapiq
import lime
import pickle
import pandas as pd
from sklearn.inspection import permutation_importance


def run_lime(model, df, target_name):

    feature_names = df.columns.drop(target_name).to_list()

    X = df.drop(columns=[target_name], axis=1).to_numpy()
    y = df[target_name]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=y.unique(),
        mode="classification" if y.nunique() < 10 else "regression",
        discretize_continuous=False,
    )

    explanations = []
    for x in X:
        expl = explainer.explain_instance(x, model.predict_proba)
        # convert the list to a dictionary
        expl_dict = {}
        for feature, value in expl.as_list():
            expl_dict[feature] = value
        explanations.append(expl_dict)

    return explanations


def run_perm(model, df, target_name):
    """This is a global method"""
    X = df.drop(columns=[target_name])
    y = df[target_name]

    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    perm_importances = result.importances_mean

    expl_dict = dict()
    for j, feature_name in enumerate(X.columns):
        expl_dict[feature_name] = perm_importances[j]
    return expl_dict


def run_shap(model, df, target_name, preds=None):
    # calculate preds if not passed
    if preds is None:
        preds = model.predict(df.drop(columns=[target_name]))

    explainer = shap.Explainer(model)

    explanations = []
    for i, row in df.reset_index(drop=True).iterrows():
        p = preds[i]
        expl = explainer.shap_values(row.drop(target_name))
        expl = expl[:, p]

        expl_dict = {}
        for j, feature_name in enumerate(df.drop(columns=[target_name]).columns):
            expl_dict[feature_name] = expl[j]
        explanations.append(expl_dict)
    return explanations


def run_shapiq(model, df, target_name, preds=None):
    n_features = df.shape[1] - 1  # excluding target variable
    explainer = shapiq.TabularExplainer(
        model=model,  # insert the model to be explained
        data=df.drop(columns=[target_name]).to_numpy(),  # insert the background data
        index="SII",
        max_order=2,
        random_state=42,
    )

    explanations = []
    for _, row in df.reset_index(drop=True).iterrows():
        x = row.drop(target_name)
        expl = explainer.explain(x.to_numpy(), budget=2 * n_features, random_state=42)
        per_feature_expl = expl.get_n_order(1).dict_values

        expl_dict = {}
        for j, feature_name in enumerate(df.drop(columns=[target_name]).columns):
            expl_dict[feature_name] = per_feature_expl[(j,)]
        explanations.append(expl_dict)

    return explanations


def contrast_explainers(model, df, target_name, tol=0.1):
    # collect preds
    preds = model.predict(df.drop(columns=[target_name]))

    # run LIME
    print("running LIME")
    lime_expls = run_lime(model, df, target_name)

    # run Permutation
    # print("running PERM")
    # perm_expls = run_perm(model, df, target_name)

    # run SHAP
    print("running SHAP")
    shap_expls = run_shap(model, df, target_name, preds=preds)

    # run SHAPIQ
    # print("running SHAPIQ")
    # shapiq_expls = run_shapiq(model, df, target_name, preds=preds)

    # add preds to df
    df["prediction"] = preds

    expl_diffs = []
    # for i, (shap_expl, lime_expl, perm_expl) in enumerate(zip(shap_expls, lime_expls, perm_expls)):
    # for i, (shap_expl, lime_expl, shapiq_expl) in enumerate(
    #     zip(shap_expls, lime_expls, shapiq_expls)
    # ):
    for i, (shap_expl, lime_expl) in enumerate(zip(shap_expls, lime_expls)):
        diff = {"record": df.iloc[i]}

        for feature in lime_expl:
            for other_expl in [lime_expl]:  # , shapiq_expl]:  # , perm_expl]:
                # consider important diffs...
                if not (abs(shap_expl[feature] - other_expl[feature]) > tol):
                    continue

                # ...that have opposite signs
                if shap_expl[feature] * other_expl[feature] > 0:
                    continue

                # then store the difference
                diff[feature] = {
                    "lime": lime_expl[feature],
                    "shap": shap_expl[feature],
                    # "shapiq": shapiq_expl[feature],
                    # "perm": perm_expl[feature],
                }
        # if diff has something more than the record
        if len(diff) > 1:
            expl_diffs.append(diff)

    return expl_diffs


if __name__ == "__main__":
    # load model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # load valid data
    df = pd.read_csv("valid.csv")

    subset_of_indices = range(10)

    df = df.iloc[subset_of_indices]

    # run_lime2(model, df, "income")

    print(contrast_explainers(model, df, "income"))
