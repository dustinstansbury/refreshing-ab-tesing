import numpy as np
import scipy as sp
import pandas as pd

import streamlit as st
import holoviews as hv

from collections import OrderedDict

from spearmint.experiment import Experiment
from spearmint.hypothesis_test import HypothesisTest, HypothesisTestGroup
from spearmint.stats import Samples
from spearmint import vis
from spearmint.utils import infer_variable_type

from vis import (
    CONTROL_COLOR,
    get_variation_color,
    visualize_means_delta_results,
    visualize_proportions_delta_results,
    visualize_rates_ratio_results,
    visualize_bootstrap_delta_results,
    visualize_bayesian_delta_results,
)


# Until streamlit supports Bokeh 3,
# we must always use matplotlib backend for plots.
# This overloads the setting spearmint.cfg::vis:vis_backend
hv.extension("matplotlib")


def reload_page():
    st.set_page_config(layout="wide")


@st.cache_data
def load_csv(dataset_file):
    return pd.read_csv(dataset_file)


def get_list_index(lst, entry):
    if entry in lst:
        return lst.index(entry)

    return 0


def get_samples(df, metric_column, treatment_column, variation_name):
    values = df.loc[df[treatment_column] == variation_name, metric_column]
    return Samples(values, name=variation_name)


def load_dataset(dataset_file):
    if dataset_file is None:
        dataset_file = st.session_state.get("dataset_file")

    if dataset_file is not None:
        dataframe = load_csv(dataset_file)
        st.session_state["dataset_file"] = dataset_file
        st.session_state["dataframe"] = dataframe
        st.session_state["data_columns"] = dataframe.columns.tolist()
        st.session_state["available_data_columns"] = dataframe.columns.tolist()


def plot_samples(control_samples, variation_samples, variable_type):
    if variable_type == "binary":
        layout = vis.plot_bernoulli(
            p=control_samples.mean,
            label=f"{control_samples.name} (control)",
            color=CONTROL_COLOR,
        )
        for ii, vs in enumerate(variation_samples):
            layout *= vis.plot_bernoulli(
                p=vs.mean,
                label=vs.name,
                color=get_variation_color(ii),
            )
    else:
        layout = vis.plot_kde(
            samples=control_samples.data,
            label=f"{control_samples.name} (control)",
            color=CONTROL_COLOR,
        )
        for ii, vs in enumerate(variation_samples):
            layout *= vis.plot_kde(
                samples=vs.data,
                # std=vs.std,
                label=vs.name,
                color=get_variation_color(ii),
            )

    return layout.relabel("Dataset Samples (KDE)").opts(legend_position="right")


def plot_results():
    if st.session_state["inference_method"] == "frequentist":
        if st.session_state["variable_type"] == "binary":
            layout = visualize_proportions_delta_results(
                st.session_state["test_results"]
            )
        elif st.session_state["variable_type"] == "continuous":
            layout = visualize_means_delta_results(st.session_state["test_results"])
        elif st.session_state["variable_type"] == "counts":
            layout = visualize_rates_ratio_results(st.session_state["test_results"])
    elif st.session_state["inference_method"] == "bootstrap":
        layout = visualize_bootstrap_delta_results(st.session_state["test_results"])
    elif st.session_state["inference_method"] == "bayesian":
        layout = visualize_bayesian_delta_results(st.session_state["test_results"])

    return st.write(hv.render(layout))


def get_mc_correction():
    MULTIPLE_COMPARISON_METHODS = (
        "sidak",
        "bonferroni",
        "fdr_bh",
    )
    mc_correction_method = st.selectbox(
        "Multiple Comparison Method",
        MULTIPLE_COMPARISON_METHODS,
        index=MULTIPLE_COMPARISON_METHODS.index(
            st.session_state.get("mc_correction_method", "sidak")
        ),
    )
    st.session_state["mc_correction_method"] = mc_correction_method
    return mc_correction_method


def get_bayesian_model_choices():
    if st.session_state["variable_type"] == "counts":
        return ["poisson"]
    elif st.session_state["variable_type"] == "binary":
        return ["bernoulli", "binomial"]
    else:
        return ["gaussian", "student_t"]


def get_bayesian_parameter_estimation_choices():
    if st.session_state["bayesian_model_name"] == "bernoulli":
        return ["analytic", "mcmc", "advi"]
    elif st.session_state["bayesian_model_name"] == "binomial":
        return ["analytic", "mcmc"]
    elif st.session_state["bayesian_model_name"] == "gaussian":
        return ["analytic", "mcmc", "advi"]
    elif st.session_state["bayesian_model_name"] == "student_t":
        return ["mcmc", "advi"]
    elif st.session_state["bayesian_model_name"] == "poisson":
        return ["analytic", "mcmc"]


def run_inference():
    alpha = st.session_state["alpha"]
    variation_groups = st.session_state["variation_groups"]
    n_tests = len(variation_groups)
    experiment = Experiment(st.session_state.dataframe)
    tests = []
    test_results = []
    for ii, vg in enumerate(variation_groups):
        st.info(
            f"Running test {ii+1}/{n_tests}. ({st.session_state.control_group} vs {vg}, alpha={alpha})",
            icon="🤖",
        )
        test = HypothesisTest(
            control=st.session_state["control_group"],
            variation=vg,
            metric=st.session_state["metric_column"],
            treatment=st.session_state["treatment_column"],
            variable_type=st.session_state["variable_type"],
            hypothesis=st.session_state["hypothesis"],
            inference_method=st.session_state["inference_method"],
            **st.session_state["inference_params"],
        )
        tests.append(test)
        test_results.append(experiment.run_test(test, alpha=alpha))

    mc_correction_method = st.session_state["inference_params"].get(
        "mc_correction_method"
    )
    if mc_correction_method is not None:
        test_group = HypothesisTestGroup(
            tests=tests,
            correction_method=mc_correction_method,
        )
        st.info(
            f"Running Multiple-comparison correction using `{mc_correction_method}` method.",
            icon="🤖",
        )
        test_group_results = experiment.run_test_group(test_group, alpha=alpha)
        test_results = test_group_results.corrected_results

    st.session_state["test_results"] = test_results
    test_results_df = pd.concat(
        [tr.to_dataframe() for tr in test_results],
        axis=0,
    )
    test_results_df.index = variation_groups
    test_results_df.index.name = "variation"
    st.session_state["test_results_df"] = test_results_df


reload_page()

# -------- Dataset Specification -----------

"""# ✨ Refreshing Hypothesis Testing ✨

To use this app

1. 📁 **Import a Dataset** csv file
2. 🔍 **Specify the data columns** that define the **metric** and **treatment groups**
3. 💡**Specify your Hypothesis**, including comparison type and acceptable alpha
4. 🛠️ **Configure the Inference Procedure**. You can use **frequentist**, **bootstrap**, or **Bayesian** inference methods.
5. ⚡️ **Run the Analysis** and interpret the results
"""

"""## 📁 Dataset"""
dataset_file = st.file_uploader(
    label="Choose a dataset file to import",
    type=["csv", "tsv"],
)

load_dataset(dataset_file)

if st.session_state.get("dataframe") is not None:
    st.dataframe(st.session_state.dataframe)

    if st.session_state.get("available_data_columns") is not None:
        """#### 🔍 Specify Data Columns"""
        mcol, vtcol = st.columns(2)

        with mcol:
            st.session_state["metric_column"] = metric_column = st.selectbox(
                "Metric Column",
                [st.session_state.get("metric_column", None)]
                + st.session_state["available_data_columns"],
            )

        inferred_variable_type = "continuous"
        if st.session_state.get("metric_column") is not None:
            inferred_variable_type = infer_variable_type(
                st.session_state.dataframe[metric_column]
            )
        VARIABLE_TYPES = ("binary", "continuous", "counts")
        with vtcol:
            variable_type = st.selectbox(
                "Metric Variable Type",
                VARIABLE_TYPES,
                index=VARIABLE_TYPES.index(inferred_variable_type),
            )
            st.session_state["variable_type"] = variable_type

        tcol, _ = st.columns(2)
        with tcol:
            st.session_state["treatment_column"] = treatment_column = st.selectbox(
                "Treatment Column",
                [st.session_state.get("treatment_column", None)]
                + st.session_state["available_data_columns"],
            )

        if treatment_column is not None:
            ccol, vcol = st.columns(2)

            treatment_columns = sorted(
                st.session_state["dataframe"][treatment_column].unique()
            )

            old_control_group = st.session_state.get("control_group")
            with ccol:
                st.session_state["control_group"] = control_group = st.selectbox(
                    "Control Treatment",
                    treatment_columns,
                    index=get_list_index(
                        treatment_columns, st.session_state.get("control_group")
                    ),
                )

            available_variation_groups = [
                c for c in treatment_columns if c != control_group
            ]
            # Update, so reset
            if control_group != old_control_group:
                default_variation_groups = available_variation_groups
            else:
                default_variation_groups = st.session_state.get("variation_groups")
                default_variation_groups = (
                    default_variation_groups
                    if default_variation_groups is not None
                    else available_variation_groups
                )
            with vcol:
                st.session_state[
                    "variation_groups"
                ] = variation_groups = st.multiselect(
                    "Variation Treatment(s)",
                    available_variation_groups,
                    default=default_variation_groups,
                )
            st.session_state["n_variations"] = len(variation_groups)

            control_samples = get_samples(
                st.session_state.dataframe,
                st.session_state.metric_column,
                st.session_state.treatment_column,
                st.session_state.control_group,
            )
            variation_samples = []
            for treatment_name in st.session_state["variation_groups"]:
                variation_samples.append(
                    get_samples(
                        st.session_state.dataframe,
                        st.session_state.metric_column,
                        st.session_state.treatment_column,
                        treatment_name,
                    )
                )

            st.write(
                hv.render(
                    plot_samples(control_samples, variation_samples, variable_type)
                )
            )

    # -------- Hypothesis Specification -----------

    """## 💡 Hypothesis"""
    HYPOTHESIS_OPTIONS = OrderedDict(
        [
            ("Variation is larger than control", "larger"),
            ("Variation is smaller control", "smaller"),
            ("Conditions are unequal", "unequal"),
        ]
    )

    hcol1, hcol2 = st.columns(2)

    with hcol1:
        st.session_state["hypothesis"] = HYPOTHESIS_OPTIONS[
            st.selectbox(label="Hypothesis", options=HYPOTHESIS_OPTIONS.keys())
        ]

    with hcol2:
        st.session_state["alpha"] = st.slider(
            "alpha (Acceptable False Positive Rate)",
            min_value=0.01,
            max_value=0.5,
            step=0.001,
            value=0.05,
            format="%1.3f",
            label_visibility="visible",
        )

    # -------- Inference Specification -----------

    """## 🛠️ Inference Procedure"""

    icol, ocol = st.columns(2)

    INFERENCE_METHODS = ("frequentist", "bootstrap", "bayesian")

    with icol:
        st.session_state["inference_method"] = inference_method = st.selectbox(
            label="""Inference Method
            """,
            options=INFERENCE_METHODS,
            index=INFERENCE_METHODS.index(
                st.session_state.get("inference_method", "frequentist")
            ),
        )

    n_variations = st.session_state.get("n_variations", 1)

    inference_params = {}
    with ocol:
        if inference_method == "frequentist":
            if n_variations > 1:
                inference_params["mc_correction_method"] = get_mc_correction()
        elif inference_method == "bootstrap":
            if n_variations > 1:
                inference_params["mc_correction_method"] = get_mc_correction()

            def compile_statistic_function(function_text):
                func = eval(function_text)
                try:
                    func([1, 2, 3])
                except Exception as e:
                    st.error(f"Invalid statistic function: {e}")
                return func

            statistic_function_text = st.text_input(
                """Define a custom statistic function for the bootstrap.
                You can use numpy (np) or scipy (sp) functions in the
                definition
                """,
                value="""np.mean""",
            )
            statistic_function = compile_statistic_function(statistic_function_text)
            inference_params["statistic_function"] = statistic_function

        else:
            # model
            model_choices = get_bayesian_model_choices()

            st.session_state[
                "bayesian_model_name"
            ] = bayesian_model_name = st.selectbox(
                "Bayesian Model",
                model_choices,
                index=0,
            )
            inference_params["bayesian_model_name"] = bayesian_model_name

            # parameter estimation method
            parameter_estimation_choices = get_bayesian_parameter_estimation_choices()

            st.session_state[
                "bayesian_parameter_estimation_method"
            ] = bayesian_parameter_estimation_method = st.selectbox(
                "Parameter Estimation Method",
                parameter_estimation_choices,
                index=0,
            )
            inference_params[
                "bayesian_parameter_estimation_method"
            ] = bayesian_parameter_estimation_method
        st.session_state["inference_params"] = inference_params

    # -------- Analysis -----------

    """## ⚡️ Analysis"""

    run_analysis = st.button(label="Run Analysis")
    if run_analysis:
        run_inference()

        """#### 📊 Test Results"""
        plot_results()

        rcol1, rcol2 = st.columns([0.3, 0.7])

        with rcol1:
            """#### Test Summary"""
            st.dataframe(
                st.session_state["test_results_df"][["hypothesis", "accept_hypothesis"]]
            )

        with rcol2:
            """#### Details"""
            st.dataframe(st.session_state["test_results_df"])


else:
    st.write("No dataset specified, please load one from a file.")


# floating footer
footer = """<style>
a:link , a:visited{
    color: #13A085;
    background-color: transparent;
}

a:hover,  a:active {
    color: #01CBA6;
    background-color: transparent;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: lightgray;
    text-align: right;
}
</style>
<div class="footer">
    Powered by <a href="https://github.com/dustinstansbury/spearmint" target="_blank">spearmint</a>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
