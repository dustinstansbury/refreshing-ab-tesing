import warnings
import numpy as np
import scipy as sp
import pandas as pd

import streamlit as st
import holoviews as hv
from holoviews.plotting.mpl import MPLPlot

from collections import OrderedDict

from spearmint.experiment import Experiment
from spearmint.hypothesis_test import HypothesisTest, HypothesisTestGroup
from spearmint.stats import Samples
from spearmint.utils import infer_variable_type, format_value

from vis import (
    plot_samples,
    plot_frequentist_continuous_results,
    plot_frequentist_binary_results,
    plot_frequentist_counts_resuls,
    plot_bootstrap_results,
    plot_bayesian_results,
)

import doc


# Until streamlit supports Bokeh 3,
# we use matplotlib backend for plots.
# This overloads the setting spearmint.cfg::vis:vis_backend
warnings.filterwarnings("ignore", module="holoviews")
MPLPlot.sublabel_format = ""
hv.extension("matplotlib")


MAX_BOOTSTRAP_NOBS = 10_000
MAX_MCMC_NOBS = 5_000
MAX_N_VARIATIONS = 20


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
        st.session_state = {}
        # dataset_file = st.session_state.get("dataset_file")

    if dataset_file is not None:
        dataframe = load_csv(dataset_file)
        st.session_state["dataset_file"] = dataset_file
        st.session_state["dataframe"] = dataframe
        st.session_state["data_columns"] = dataframe.columns.tolist()
        st.session_state["available_data_columns"] = dataframe.columns.tolist()


def summarize_samples(samples, use_container_width):
    def _make_df(samples, is_control):
        name = samples.name
        if is_control:
            name += " (control)"
        return pd.DataFrame(
            (
                samples.nobs,
                format_value(samples.mean, precision=4),
                format_value(samples.confidence_interval(0.95), precision=3),
                format_value(samples.var, precision=4),
            ),
            index=[
                "# Observations",
                "mean",
                "95% CI",
                "variance",
            ],
            columns=[name],
        ).T

    if isinstance(samples, list):
        dfs = []
        for ii, samp in enumerate(samples):
            dfs.append(_make_df(samp, ii == 0))

        df = pd.concat(dfs, axis=0)
    else:
        df = _make_df(samples)

    st.dataframe(df, use_container_width=use_container_width)


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
        help=doc.tooltip.select_mc_correction,
    )
    st.session_state["mc_correction_method"] = mc_correction_method
    return mc_correction_method


def get_bayesian_model_choices():
    if st.session_state["variable_type"] == "counts":
        return ["poisson"]
    elif st.session_state["variable_type"] == "binary":
        return ["binomial", "bernoulli"]
    else:
        return ["gaussian", "student_t"]


def get_bayesian_parameter_estimation_choices():
    if st.session_state["bayesian_model_name"] == "bernoulli":
        return ["analytic", "mcmc"]
    elif st.session_state["bayesian_model_name"] == "binomial":
        return ["analytic", "mcmc"]
    elif st.session_state["bayesian_model_name"] == "gaussian":
        return ["analytic", "mcmc"]
    elif st.session_state["bayesian_model_name"] == "student_t":
        return ["mcmc"]
    elif st.session_state["bayesian_model_name"] == "poisson":
        return ["analytic", "mcmc"]


def run_inference():
    alpha = st.session_state["alpha"]
    variation_groups = st.session_state["variation_groups"]
    n_tests = len(variation_groups)

    experiment = Experiment(
        st.session_state["data"],
        measures=[st.session_state["metric_column"]],
        treatment=st.session_state["treatment_column"],
    )
    tests = []
    test_results = []
    for ii, vg in enumerate(variation_groups):
        st.info(
            f"Running test {ii+1}/{n_tests}. ({st.session_state['control_group']} vs {vg}, alpha={alpha})",
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


def plot_results():
    if st.session_state["inference_method"] == "frequentist":
        if st.session_state["variable_type"] == "binary":
            layout = plot_frequentist_binary_results(st.session_state["test_results"])
        elif st.session_state["variable_type"] == "continuous":
            layout = plot_frequentist_continuous_results(
                st.session_state["test_results"]
            )
        elif st.session_state["variable_type"] == "counts":
            layout = plot_frequentist_counts_resuls(st.session_state["test_results"])
    elif st.session_state["inference_method"] == "bootstrap":
        layout = plot_bootstrap_results(st.session_state["test_results"])
    elif st.session_state["inference_method"] == "bayesian":
        layout = plot_bayesian_results(st.session_state["test_results"])

    return st.write(hv.render(layout))


reload_page()

# -------- Dataset Specification -----------

st.markdown(doc.intro)

show_instructions = st.checkbox("**Show/Hide Instructions**")
if show_instructions:
    st.markdown(doc.instructions.how_to_run_test)

lcol, _, _ = st.columns(3)

with lcol:
    """## 📁 Dataset"""
    dataset_file = st.file_uploader(
        label="Choose a dataset file to import",
        type=["csv", "tsv"],
        help=doc.tooltip.file_upload,
    )

load_dataset(dataset_file)

if st.session_state.get("dataframe") is not None:
    expand_dataset_df = st.checkbox("Expand Data Table")
    st.dataframe(st.session_state["dataframe"], use_container_width=expand_dataset_df)
    st.session_state["data"] = st.session_state["dataframe"].copy()

    if st.session_state.get("available_data_columns") is not None:
        """#### 🔍 Specify Data Columns"""
        mcol, vtcol = st.columns(2)

        with mcol:
            st.session_state["metric_column"] = metric_column = st.selectbox(
                "Metric Column",
                [st.session_state.get("metric_column", None)]
                + st.session_state["available_data_columns"],
                help=doc.tooltip.select_metric_column,
            )

        inferred_variable_type = "continuous"
        if st.session_state.get("metric_column") is not None:
            # Filter out any invalid metric rows
            valid_metric_mask = st.session_state["data"][metric_column].notnull()
            n_invalid_metric = (~valid_metric_mask).sum()
            if n_invalid_metric > 0:
                st.session_state["data"] = st.session_state["data"][valid_metric_mask]
                st.warning(
                    doc.warnings.removing_invalid_metrics(
                        n_invalid_metric, metric_column
                    )
                )

            inferred_variable_type = infer_variable_type(
                st.session_state["data"][metric_column]
            )
        VARIABLE_TYPES = ("binary", "continuous", "counts")
        with vtcol:
            variable_type = st.selectbox(
                "Metric Variable Type",
                VARIABLE_TYPES,
                index=VARIABLE_TYPES.index(inferred_variable_type),
                help=doc.tooltip.select_metric_variable_type,
            )
            st.session_state["variable_type"] = variable_type

        tcol, _ = st.columns(2)
        with tcol:
            st.session_state["treatment_column"] = treatment_column = st.selectbox(
                "Treatment Column",
                [st.session_state.get("treatment_column", None)]
                + st.session_state["available_data_columns"],
                help=doc.tooltip.select_treatment_column,
            )

        if st.session_state.get(
            "treatment_column"
        ) is not None and st.session_state.get("metric_column"):
            # Filter out any invalid treatment rows
            valid_treatment_mask = st.session_state["data"][treatment_column].notnull()

            n_invalid_treatments = (~valid_treatment_mask).sum()
            if n_invalid_treatments > 0:
                st.session_state["data"] = st.session_state["data"][
                    valid_treatment_mask
                ]
                st.warning(
                    doc.warnings.removing_invalid_treatments(
                        n_invalid_treatments, treatment_column
                    )
                )

            if pd.api.types.is_numeric_dtype(
                st.session_state["data"][treatment_column].dtype
            ):
                st.session_state["data"][treatment_column] = st.session_state["data"][
                    treatment_column
                ].apply(lambda x: f"{x}")

            ccol, vcol = st.columns(2)

            treatment_columns = sorted(
                st.session_state["data"][treatment_column].unique()
            )

            old_control_group = st.session_state.get("control_group")
            with ccol:
                st.session_state["control_group"] = control_group = st.selectbox(
                    "Control Treatment",
                    treatment_columns,
                    index=get_list_index(
                        treatment_columns, st.session_state.get("control_group")
                    ),
                    help=doc.tooltip.select_control_group,
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
                    help=doc.tooltip.select_variation_groups,
                )
            st.session_state["n_variations"] = n_variations = len(variation_groups)

            if n_variations > MAX_N_VARIATIONS:
                st.warning(
                    doc.warnings.too_many_variations(n_variations, MAX_N_VARIATIONS)
                )
            else:
                control_samples = get_samples(
                    st.session_state["data"],
                    st.session_state["metric_column"],
                    st.session_state["treatment_column"],
                    st.session_state["control_group"],
                )
                variation_samples = []
                for treatment_name in st.session_state["variation_groups"]:
                    variation_samples.append(
                        get_samples(
                            st.session_state["data"],
                            st.session_state["metric_column"],
                            st.session_state["treatment_column"],
                            treatment_name,
                        )
                    )

                """### Data Summary"""
                sbcol, spcol, _ = st.columns([0.15, 0.1, 0.8])  # Better way to do this?
                with sbcol:
                    use_container_width = st.checkbox("Expand Summary Table")
                with spcol:
                    _plot_samples = st.checkbox("Plot Samples")
                sumcol, plotcol = st.columns(2)
                with sumcol:
                    summarize_samples(
                        [control_samples] + variation_samples,
                        use_container_width=use_container_width,
                    )

                with plotcol:
                    if _plot_samples:
                        st.write(
                            plot_samples(
                                control_samples, variation_samples, variable_type
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
            st.selectbox(
                label="Comparison Type",
                options=HYPOTHESIS_OPTIONS.keys(),
                help=doc.tooltip.select_comparison_type,
            )
        ]

    with hcol2:
        st.session_state["alpha"] = st.slider(
            "alpha",
            min_value=0.01,
            max_value=0.5,
            step=0.001,
            value=0.05,
            format="%1.3f",
            label_visibility="visible",
            help=doc.tooltip.set_alpha,
        )

    # -------- Inference Specification -----------

    INFERENCE_METHODS = ["frequentist", "bootstrap", "bayesian"]

    """## 🛠️ Inference Procedure"""

    if (st.session_state.get("metric_column") is not None) and (
        st.session_state.get("treatment_column") is not None
    ):
        st.session_state["max_treatment_nobs"] = max_treatment_nobs = (
            st.session_state["data"]
            .groupby(st.session_state["treatment_column"])
            .count()[st.session_state["metric_column"]]
            .max()
        )

        if max_treatment_nobs > MAX_BOOTSTRAP_NOBS:
            st.warning(
                doc.warnings.too_many_treatment_nobs_for_bootstrap(
                    max_treatment_nobs, MAX_BOOTSTRAP_NOBS
                )
            )
            INFERENCE_METHODS.remove("bootstrap")

    icol, ocol = st.columns(2)

    with icol:
        st.session_state["inference_method"] = inference_method = st.selectbox(
            label="""Inference Method
            """,
            options=INFERENCE_METHODS,
            index=INFERENCE_METHODS.index(
                st.session_state.get("inference_method", "frequentist")
            ),
            help=doc.tooltip.select_inference_method,
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
                label="Statistic Function",
                value="""np.mean""",
                help=doc.tooltip.define_statistic_function,
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
                help=doc.tooltip.select_bayesian_model,
            )
            inference_params["bayesian_model_name"] = bayesian_model_name

            # parameter estimation method
            parameter_estimation_choices = get_bayesian_parameter_estimation_choices()
            if st.session_state["max_treatment_nobs"] > MAX_MCMC_NOBS:
                st.warning(
                    doc.warnings.too_many_nobs_for_mcmc(
                        max_treatment_nobs, MAX_MCMC_NOBS
                    )
                )

                parameter_estimation_choices.remove("mcmc")

            st.session_state[
                "bayesian_parameter_estimation_method"
            ] = bayesian_parameter_estimation_method = st.selectbox(
                "Parameter Estimation Method",
                parameter_estimation_choices,
                index=0,
                help=doc.tooltip.select_bayesian_parameters_estimation_method,
            )
            inference_params[
                "bayesian_parameter_estimation_method"
            ] = bayesian_parameter_estimation_method
        st.session_state["inference_params"] = inference_params

    # -------- Analysis -----------

    """## ⚡️ Analysis"""

    run_analysis = st.button(label="Run Analysis", help=doc.tooltip.run_analysis)
    show_interpretation_instructions = st.checkbox(
        "Show how to interpret Test Results",
        value=True,
        help=doc.tooltip.show_interpretations,
    )
    if run_analysis:
        run_inference()

        """### 📊 Test Results"""
        plot_results()

        rcol1, rcol2 = st.columns([0.2, 0.8])

        with rcol1:
            """### Test Summary"""
            summary = st.session_state["test_results_df"][
                ["hypothesis", "accept_hypothesis"]
            ]
            summary.loc[:, "accept_hypothesis"] = summary["accept_hypothesis"].apply(
                lambda x: "✅" if x else "❌"
            )
            st.dataframe(summary)

        with rcol2:
            """#### Details"""
            st.dataframe(st.session_state["test_results_df"])

        if show_interpretation_instructions:
            icol, _ = st.columns(2)
            with icol:
                st.write(
                    doc.instructions.get_interpretation(
                        st.session_state["inference_method"],
                        st.session_state["variable_type"],
                    )
                )


else:
    st.write("No dataset specified, please load one from a file.")


footer = """

---
##### Please report any bugs or issues to the [Issue Tracker](https://github.com/dustinstansbury/refreshing-ab-testing/issues)

<style>
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
    padding-right: 80px;
}
</style>
<div class="footer">
    Powered by <a href="https://github.com/dustinstansbury/spearmint" target="_blank">spearmint</a>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
