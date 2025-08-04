import pandas as pd
from sklearn.metrics import f1_score
import streamlit as st

def _compute_opportunity(y_true, y_pred):
    try:
        target_val = st.session_state.target_class
        mask = y_true == target_val
        oppo_preds = y_pred[mask]
        return (oppo_preds == target_val).mean()
    except ValueError as e:
        return 0


def _compute_metric_parity(
    metric_func, metric_name, df, preds, fair_columns, target_name
):
    parity = {}
    base_score = metric_func(df[target_name], preds)
    parity["baseline"] = base_score

    for col in fair_columns:

        # if numerical column
        if (
            df[col].dtype in [int, float]
            and len(df[col].unique()) > 10
        ):
            # create 3 bins
            min_val = df[col].min()
            max_val = df[col].max()
            bins = [min_val, (min_val + max_val) / 2, max_val]
            # create labels
            labels = [f"{col}={bins[i]}-{bins[i+1]}" for i in range(2)]
            # show results for the 3 bins
            for i in range(2):
                mask = (df[col] >= bins[i]) & (df[col] < bins[i + 1])
                metric_score = metric_func(df[target_name][mask], preds[mask])
                parity[labels[i]] = metric_score
            continue

        # else, if categorical
        values = sorted(df[col].unique().tolist())
        if values == [False, True]:
            values = [0, 1]

        for value in values:
            # skip 0 in OOH
            if values == [0, 1] and ("is" in col or "=" in col) and value == 0: 
                continue
            mask = df[col] == value
            score = metric_func(df[target_name][mask], preds[mask])
            parity[f"{col}={value}"] = score

    # cleanup f1_scores set to 0
    if metric_name == "f1_score_parity" or metric_name == "equal_opportunity":
        keys_to_delete = []
        for key, score in parity.items():
            if score == 0:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            parity.pop(key)

    return pd.DataFrame(parity, index=[0]).assign(metric=metric_name)


def fairness_metrics(
    df: pd.DataFrame,
    target_variable: str,
    fair_columns: list[str],
    preds: pd.Series,
) -> pd.DataFrame:

    acc_parity = _compute_metric_parity(
        lambda y_true, y_pred: (y_true == y_pred).mean(),
        "accuracy_parity",
        df,
        preds,
        fair_columns,
        target_variable,
    )
    f1_parity = _compute_metric_parity(
        f1_score, "f1_score_parity", df, preds, fair_columns, target_variable
    )
    equal_opportunity = _compute_metric_parity(
        _compute_opportunity,
        "equal_opportunity",
        df,
        preds,
        fair_columns,
        target_variable,
    )

    fairness_df = pd.concat(
        [acc_parity, f1_parity, equal_opportunity], ignore_index=True
    )
    fairness_df = fairness_df.set_index("metric")

    return fairness_df



def compute_all_group_stats(df, fair_columns, preds):
    all_group_stats = []
    for col in fair_columns:
        # determine the group and store
        if df[col].dtype in [int, float] and len(df[col].unique()) > 10:
            # create 3 bins
            min_val = df[col].min()
            max_val = df[col].max()
            bins = [min_val, (min_val + max_val) / 2, max_val]
            # create labels
            labels = [f"{col}={bins[i]}-{bins[i+1]}" for i in range(2)]
            # show results for the 3 bins
            for i in range(2):
                mask = (df[col] >= bins[i]) & (df[col] < bins[i + 1])
                group_df = df[mask]
                group_name = labels[i]
                group_stats = _get_group_stats(group_name, group_df, preds)
                all_group_stats.append(group_stats)
        else:
            # else, if categorical
            values = sorted(df[col].unique().tolist())
            if values == [False, True]:
                values = [0, 1]

            for value in values:
                # skip 0 in OOH
                if values == [0, 1] and ("is" in col or "=" in col) and value == 0:
                    continue
                mask = df[col] == value
                group_df = df[mask]
                group_name = f"{col}={value}"
                group_stats = _get_group_stats(group_name, group_df, preds)
                all_group_stats.append(group_stats)

    return all_group_stats


def _get_group_stats(group_name, group_df, preds):
    num_samples = len(group_df)
    percent_samples = num_samples / len(st.session_state.df) * 100
    percent_target_1 = (
        group_df[st.session_state.target_variable] == st.session_state.target_class
    ).mean() * 100
    percent_preds_1 = (
        preds[group_df.index] == st.session_state.target_class
    ).mean() * 100
    return {
        "group": group_name,
        "n. samples": num_samples,
        "% samples": round(percent_samples, 1),
        "% desirable target": round(percent_target_1, 1),
        "% desirable preds.": round(percent_preds_1, 1),
    }
