from collections import defaultdict
import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from llm.llm import extract_fairness_columns
from xai.calibration import get_calibration_metrics
from xai.fairness import fairness_metrics, compute_all_group_stats
from xai.attribution import contrast_explainers
from xai.simple_modeling import run_simple_modeling


def run_calibration() -> dict[str, any]:
    cm = get_calibration_metrics(
        model=st.session_state.model,
        df=st.session_state.df,
        target_variable=st.session_state.target_variable,
    )

    prob_true = cm["calib_prob_true"]
    prob_pred = cm["calib_prob_pred"]
    prob_true_iso = cm["calib_prob_true_iso"]
    prob_pred_iso = cm["calib_prob_pred_iso"]

    miscalib = pd.Series(prob_true - prob_pred).abs()
    mean_miscalib = miscalib.mean()
    max_miscalib = miscalib.max()
    miscalib_iso = pd.Series(prob_true_iso - prob_pred_iso).abs()
    mean_miscalib_iso = miscalib_iso.mean()
    max_miscalib_iso = miscalib_iso.max()

    summary_stats = {
        "miscalib": miscalib,
        "mean_misc": mean_miscalib,
        "max_misc": max_miscalib,
        "mean_misc_iso": mean_miscalib_iso,
        "max_misc_iso": max_miscalib_iso,
    }

    summary_stats.update(cm)
    return summary_stats


def display_calibration(
    calibration_col,
    calibration_tab,
    calibration_result,
    threshold_bad_mean_miscalibration: float = 0.05,
):
    explanation_container = calibration_tab.container(border=True)
    explanation_container.markdown(
        """\
💡 *What is calibration?*

The calibration of a model represents its ability to output scores \
(between 0 and 1) that reflect the true probabilities of the outcomes.
In other words, if a classifier is calibrated, we can reasonably interpret \
its predicted scores as class probabilities.
Note that a model can have high accuracy but low calibration!

📚 Some resources:
- [Wikipedia: Probabilistic classification](https://en.wikipedia.org/wiki/Probabilistic_classification)
- [Scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Wikipedia: Isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression)
"""
    )

    mean_misc = calibration_result["mean_misc"]
    max_misc = calibration_result["max_misc"]
    mean_misc_iso = calibration_result["mean_misc_iso"]
    max_misc_iso = calibration_result["max_misc_iso"]

    delta = int(round(mean_misc * 100))
    calibration_col.metric(
        label="Calibration",
        value=(
            "To Check" if mean_misc >= threshold_bad_mean_miscalibration else "Passed"
        ),
        delta=(
            str(-delta) + "%"
            if mean_misc >= threshold_bad_mean_miscalibration
            else "No issue detected"
        ),
    )

    calibration_tab.markdown(
        """\
        Assessment of calibration \
        (simulated using 50% of the dev set), including \
        with an isotonic regression layer to improve calibration:"""
    )
    summary_df = pd.DataFrame(
        {
            "metric": [
                "Mean miscalibration (%)",
                "Max miscalibration (%)",
                "Brier score",
            ],
            "original model": [
                round(mean_misc * 100, 1),
                round(max_misc * 100, 1),
                calibration_result["calib_brier"],
            ],
            "original + isotonic": [
                round(mean_misc_iso * 100, 1),
                round(max_misc_iso * 100, 1),
                calibration_result["calib_brier_iso"],
            ],
        }
    ).set_index("metric")
    calibration_tab.dataframe(summary_df)

    if (
        mean_misc > threshold_bad_mean_miscalibration
        and mean_misc_iso > threshold_bad_mean_miscalibration
    ):
        calibration_tab.write(
            f"⚠️ The model appears to be miscalibrated ({delta}% mean miscalibration) and \
            adding isotonic regression might still be insufficient."
        )
    elif mean_misc > threshold_bad_mean_miscalibration:
        calibration_tab.write(
            f"⚠️ The model appears to be miscalibrated ({delta}% mean miscalibration), however \
            fitting an isotonic model on top might help with calibration."
        )
    else:
        calibration_tab.write(
            "✅ The model appears to be decently calibrated (mean miscalibration < 5%)."
        )

    fig, ax = plt.subplots()
    sns.set_style("darkgrid")
    sns.lineplot(
        x=calibration_result["calib_prob_pred"],
        y=calibration_result["calib_prob_true"],
        ax=ax,
        label="original model",
    )
    sns.lineplot(
        x=calibration_result["calib_prob_pred_iso"],
        y=calibration_result["calib_prob_true_iso"],
        ax=ax,
        label="original+isotonic",
    )
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration (theoretical)")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("true probability")
    ax.legend()

    calib_plot_container = calibration_tab.container(border=True)
    calib_plot_container.pyplot(fig)


def run_fairness() -> dict[str, any] | None:

    # check that LLM_API_KEY and LLM_MODEL are set
    if (
        "LLM_API_KEY" not in os.environ
        or not os.environ["LLM_API_KEY"]
        or "LLM_MODEL" not in os.environ
        or not os.environ["LLM_MODEL"]
    ):
        return None

    result = {
        "fair_columns": [],
        "outcome_md": None,
        "num_cases": 0,
        "num_metrics": 0,
    }

    # call llm
    fair_msg, fair_columns = extract_fairness_columns(
        st.session_state.df,
    )
    if len(fair_columns) == 0:
        return result

    result["fair_columns"] = fair_columns

    preds = st.session_state.model.predict(
        st.session_state.df.drop(columns=[st.session_state.target_variable])
    )

    # report metrics per group
    all_group_stats = compute_all_group_stats(st.session_state.df, fair_columns, preds)

    result["group_stats"] = pd.DataFrame(all_group_stats).set_index("group")

    fair_df = fairness_metrics(
        st.session_state.df,
        st.session_state.target_variable,
        fair_columns,
        preds,
    )

    problematic_cases = defaultdict(list)
    baseline_scores = {}
    for metric, row in fair_df.iterrows():
        base_score = row["baseline"]
        baseline_scores[metric] = base_score
        # for each other column in the Series (row):
        for column, value in row.items():
            if column == "baseline":
                continue
            if (base_score - value) > 0.05:
                problematic_cases[metric].append((column, value))

    result["num_metrics"] = len(problematic_cases.keys())
    result["num_cases"] = len(problematic_cases)
    result["problematic_groups"] = set()

    if len(problematic_cases) > 0:

        result_dict = {
            "metric": [],
            "group name": [],
            "baseline score": [],
            "group score": [],
            "disparity %": [],
        }

        for metric_name, list_of_pcs in problematic_cases.items():
            baseline_score = round(baseline_scores[metric_name], 3)
            for pc_name, pc_score in list_of_pcs:
                result_dict["metric"].append(metric_name)
                result_dict["baseline score"].append(baseline_score)
                result_dict["group name"].append(pc_name)
                result_dict["group score"].append(pc_score)
                result_dict["disparity %"].append(
                    round(abs(baseline_score - pc_score) / baseline_score * 100)
                )
                result["problematic_groups"].add(pc_name)
        result["outcome_df"] = pd.DataFrame(result_dict)

    result["problematic_groups"] = list(result["problematic_groups"])

    return result


def display_fairness(fairness_col, fairness_tab, fairness_result):

    explanation_container = fairness_tab.container(border=True)
    explanation_container.markdown(
        """\
💡 *What is fairness?*

Fairness concerns the model's ability to make unbiased and equitable predictions across different groups of individuals.
This means that the model should not favor one group over another, and its performance should be consistent across all demographic groups.
Fairness is particularly important in sensitive applications, such as hiring, lending, and healthcare, where biased predictions can have serious consequences.

📚 Some resources:
- [Wikipedia: Fairness (Machine Learning)](https://en.wikipedia.org/wiki/Fairness_(machine_learning))
- [Barocas, Hardt, Narayan 2023: Fairness and Machine Learning: Limitations and Opportunities](https://fairmlbook.org/)
"""
    )

    if fairness_result is None:
        fairness_col.metric(
            label="Fairness",
            value="Skipped",
            delta="No issue detected",
        )
        fairness_tab.write(
            "⚠️ Fairness checks require the use of an LLM. An LLM API key and/or LLM model could not be found in the environment variables."
        )
        return

    # tot_cases = len(fairness_result["fair_columns"]) * fairness_result["num_metrics"]
    # if tot_cases == 0:
    #    tot_cases = 1
    # bad_cases = fairness_result["num_cases"]
    tot_cases = len(fairness_result["fair_columns"])
    bad_cases = len(fairness_result["problematic_groups"])
    delta = round(bad_cases / tot_cases * 100)

    fairness_col.metric(
        label="Fairness",
        value=("To Check" if fairness_result["num_cases"] > 0 else "Passed"),
        delta=(str(-delta) + "%" if delta > 0 else "No issue detected"),
    )

    if len(fairness_result["fair_columns"]) > 0:
        # fairness_tab.write(
        #    "The following features might be used to discriminate groups"
        # )
        # fairness_tab.write(fairness_result["fair_columns"])
        fairness_tab.write(
            "🔍 Identified features representing groups that could be \
                discriminated (and salient statistics):"
        )
        fairness_tab.write(fairness_result["group_stats"])

        if fairness_result["num_cases"] > 0:
            fairness_tab.write(
                f"⚠️ Disparities found between the whole population (baseline) \
                and the following groups ({delta}% of the groups):"
            )
            # fairness_tab.markdown(fairness_result["outcome_md"])
            fairness_tab.write(fairness_result["outcome_df"].set_index("metric"))
        else:
            fairness_tab.write("No substantial disparities detected.")
    else:
        fairness_tab.write("No features to discriminate groups detected.")


def run_attribution(subsample_no: int | None = None) -> dict[str, any]:
    result = {
        "attribution_diffs": [],
        "outcome_df": None,
    }

    df_to_use = st.session_state.df
    if subsample_no and subsample_no > 0:  # on a subset of the data
        df_to_use = st.session_state.df.sample(subsample_no)

    # contrast explainers
    attribution_diffs = contrast_explainers(
        st.session_state.model,
        df_to_use,
        st.session_state.target_variable,
    )

    if attribution_diffs:
        result["attribution_diffs"] = attribution_diffs
        outcome_msg = "Inconsistencies found in feature attribution"
        if subsample_no:
            outcome_msg += f" (on a sample of {subsample_no} records)"

        # assemble df from diffs
        info = defaultdict(list)
        for diff in attribution_diffs:
            problematic_record = diff["record"]
            record_idx = problematic_record.name
            features = list(diff.keys())
            features.remove("record")
            for feature_name in features:
                lime_expl = round(diff[feature_name]["lime"] * 100, 1)
                shap_expl = round(diff[feature_name]["shap"] * 100, 1)
                # shapiq_expl = round(diff[feature_name]["shapiq"] * 100, 1)
                # perm_expl = round(diff[feature_name]["perm"] * 100, 1)

                # make lime and shap expl str
                lime_expl = str(lime_expl)
                shap_expl = str(shap_expl)
                # shapiq_expl = str(shapiq_expl)
                # perm_expl = str(perm_expl)
                if lime_expl[0] != "-":
                    lime_expl = "+" + lime_expl
                if shap_expl[0] != "-":
                    shap_expl = "+" + shap_expl
                # if shapiq_expl[0] != "-":
                #    shapiq_expl = "+" + shapiq_expl
                # if perm_expl[0] != "-":
                #    perm_expl = "+" + perm_expl

                info["record idx"].append(record_idx)
                info["label"].append(
                    problematic_record[st.session_state.target_variable]
                )
                info["prediction"].append(problematic_record["prediction"])
                info["feature"].append(feature_name)
                info["LIME attr. (%)"].append(lime_expl)
                info["SHAP attr. (%)"].append(shap_expl)
                # info["SHAPIQ attr. (%)"].append(shapiq_expl)
                # info["Perm. attr. (%)"].append(perm_expl)
        info_df = pd.DataFrame(info).set_index("record idx")
        info_df.sort_values(by=["record idx", "feature"], inplace=True)
        result["outcome_df"] = info_df

    return result


def display_attribution(
    attribution_col,
    attribution_tab,
    attribution_result,
    subsample_no: int | None = None,
):

    explanation_container = attribution_tab.container(border=True)
    explanation_container.markdown(
        """\
💡 *What is feature attribution?*

Feature attribution means quantifying the contribution of each feature (or feature interaction) to the model's predictions.
It can be used to explain, to some extent, the model's behavior and decision-making process.
Multiple feature attribution algorithms exist, each typically making some assumptions and approximations.
Unless there is a clear understanding of these aspects,
trusting feature attribution explanations without scrutiny can be misleading.

📚 Some resources:
- [Molnar 2025: Intepretable Machine Learning - LIME](https://christophm.github.io/interpretable-ml-book/lime.html)
- [Molnar 2025: Intepretable Machine Learning - SHAP](https://christophm.github.io/interpretable-ml-book/shap.html)
- [SHAP-IQ Docs: What Are Shapley Interactions, and Why Should You Care?](https://shapiq.readthedocs.io/en/latest/introduction/index.html)
"""
    )

    num = len(attribution_result["outcome_df"].index.unique())
    denom = len(st.session_state.df) if subsample_no is None else subsample_no
    delta = int(round(num / denom * 100))
    attribution_col.metric(
        label="Attribution",
        value=(
            "To Check" if len(attribution_result["attribution_diffs"]) > 0 else "Passed"
        ),
        delta=str(-delta) + "%" if delta > 0 else "No issue detected",
    )

    # attr_initial_txt = """\
    # Comparison of feature attribution using \
    # local linear approximation ([LIME](https://christophm.github.io/interpretable-ml-book/lime.html)), \
    # Shapley value approximation ([SHAP](https://christophm.github.io/interpretable-ml-book/shap.html)), and \
    # 2nd order Shapley interaction approximation ([SHAPIQ](https://github.com/InterpretML/shapley-importance-quantification)) \
    # """
    attr_initial_txt = """\
    Comparison of feature attribution using \
    local linear approximation ([LIME](https://christophm.github.io/interpretable-ml-book/lime.html)) and \
    Shapley value approximation ([SHAP](https://christophm.github.io/interpretable-ml-book/shap.html))\
"""

    if subsample_no is not None:
        attr_initial_txt += f" on a sample of {subsample_no} records."
    else:
        attr_initial_txt += "."
    attribution_tab.markdown(attr_initial_txt)

    if len(attribution_result["attribution_diffs"]) > 0:
        msg = f"⚠️ Relevant inconsistencies found among feature attribution \
            methods in {delta}% of the analyzed records:"
        attribution_tab.write(msg)
        attribution_tab.write(attribution_result["outcome_df"])
    else:
        attribution_tab.write(
            "✅ No substantial inconsistencies found among feature attribution methods."
        )


def run_simpler_model() -> dict[str, any]:
    """Calculates the accuracy, balanced_accuracy, and F1 score of the model first.
    Then runs logistic regression (LR) and EBM with hyper-param tuning in cross-val and computes its metrics too.
    It then compares the avg. scores of best-performing params for LR against the original model.
    Finally, it returns these outcomes.
    """
    X = st.session_state.df.drop(columns=[st.session_state.target_variable])
    y = st.session_state.df[st.session_state.target_variable]

    comparison_df = run_simple_modeling(X, y, st.session_state.model)
    print(comparison_df)

    # Determine if simpler_model model is feasible
    def within_5_percent(original, simpler_model):
        return abs(original - simpler_model) / original <= 0.05

    original_metrics = comparison_df["original model"]
    lr_metrics = comparison_df["logistic regression (est.)"]
    ebm_metrics = comparison_df["explainable boosting (est.)"]

    comparison_list = [
        (
            within_5_percent(original_metrics.loc[metric], lr_metrics.loc[metric])
            or within_5_percent(original_metrics.loc[metric], ebm_metrics.loc[metric])
        )
        for metric in ["accuracy", "balanced accuracy", "F1 score"]
    ]
    comparison_df["within 5% (rel.)"] = comparison_list

    result = dict()
    result["comparison_df"] = comparison_df
    result["comparison_list"] = comparison_list
    result["simpler_model_feasible"] = any(comparison_list)
    result["num_matched_metrics"] = sum(comparison_list)

    percent_prob = int(
        round(result["num_matched_metrics"] / len(result["comparison_list"]) * 100)
    )

    if result["simpler_model_feasible"]:
        result["msg"] = (
            f"⚠️ A simple model ([logistic regression](https://interpret.ml/docs/lr.html) or \
            [explainable boosting machine](https://interpret.ml/docs/ebm.html), estimation using 3-fold cross-val on dev. set) performs \
            comparably to the original model in {percent_prob}% of the tested metrics:"
        )
    else:
        result["msg"] = (
            "✅ A simple model ([logistic regression](https://interpret.ml/docs/lr.html) or\
            [explainable boosting machine](https://interpret.ml/docs/ebm.html), estimation using 3-fold cross-val on dev. set) seems \
            unable to match the performance of the original model."
        )

    return result


def display_simpler_model(simpler_model_col, simpler_model_tab, simpler_model_result):

    explanation_container = simpler_model_tab.container(border=True)
    explanation_container.markdown(
        """\
💡 *Why attempting to use simpler models?*

Certain black-box model classes (e.g. for tabular data `xgboost`, `catboost`, `lightgbm`)
perform very well in many different scenarios: making them a popular choice. 

However, there are cases where simpler and inherently-interpretable models \
(e.g., `linear/logistic regression`, `single tree`, `symbolic regression`) can \
perform also well.
Certain regulatory guidelines require a justification
for the rationale behind model choice: it is therefore important to test \
whether the use of a complex and opaque model over a simpler one is warranted.

📚 Some resources:
- [Rudin 2019: Stop explaining black box models for high stakes decisions and use intepretable models instead](https://doi.org/10.1038/s42256-019-0048-x)
- [Christodoulou et al. 2019: A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models](https://doi.org/10.1016/j.jclinepi.2019.02.004)
- [European Union 2024: The EU AI Act](https://www.europarl.europa.eu/topics/en/article/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence)
"""
    )

    delta = simpler_model_result["num_matched_metrics"] / len(
        simpler_model_result["comparison_list"]
    )
    delta = int(round(delta * 100))
    simpler_model_col.metric(
        label="Simpler modeling",
        value=(
            "To Check" if simpler_model_result["simpler_model_feasible"] else "Passed"
        ),
        delta=str(-delta) + "%" if delta > 0 else "No issue detected",
    )
    simpler_model_tab.write(simpler_model_result["msg"])
    simpler_model_tab.dataframe(simpler_model_result["comparison_df"])
