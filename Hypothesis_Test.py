import warnings
import numpy as np
import scipy as sp
import pandas as pd

import streamlit as st
import holoviews as hv

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


# Until streamlit supports Bokeh 3,
# we use matplotlib backend for plots.
# This overloads the setting spearmint.cfg::vis:vis_backend
warnings.filterwarnings("ignore", module="holoviews")
hv.extension("matplotlib")

MAX_BOOTSTRAP_NOBS = 10_0000


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


def summarize_samples(samples):
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

    st.dataframe(df, use_container_width=True)


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
        help="""When using Frequentist or Bootstrap inference for multiple
        variation groups, one must perform [Multiple Comparison correction](https://en.wikipedia.org/wiki/Multiple_comparisons_problem) to
        avoid inflated Type I error rate. Select your correction method here. If
        you're unsure about which correction method to use, you can always stick with
        the default method 😉.
        """,
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

    experiment = Experiment(
        st.session_state["data"],
        measures=[st.session_state["metric_column"]],
        treatment=st.session_state["treatment_column"],
    )
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

st.markdown(
    """
# Easy AB Testing

_Powered by [spearmint](https://github.com/dustinstansbury/spearmint)_

Use this simple app to run AB tests using your own datasets. The app
supports many different inference methods and variable types.

To run a test, you'll need to:

1. 📁 **Import a Dataset** csv file from you computer.
2. 🔍 **Specify the data columns** that define the **metric** and **treatment groups** in your study.
3. 💡**Specify your Hypothesis**, including comparison type and acceptable alpha. Or, just use the defaults.
4. 🛠️ **Configure the Inference Procedure**. You can use **frequentist**, **bootstrap**, or **Bayesian** inference methods. Or, just use the defaults.
5. ⚡️ **Run the Analysis** and interpret the results
"""
)

"""## 📁 Dataset"""
dataset_file = st.file_uploader(
    label="Choose a dataset file to import",
    type=["csv", "tsv"],
    help="Select a dataset file. The dataset must contain at least two columns, "
    "with one of the columns defining the values of a `metric` used for comparison. "
    "The other column should define a set of discrete values used for defining "
    "the control and variation `treatment`s.",
)

load_dataset(dataset_file)

if st.session_state.get("dataframe") is not None:
    st.dataframe(st.session_state.dataframe)
    st.session_state["data"] = st.session_state["dataframe"].copy()

    if st.session_state.get("available_data_columns") is not None:
        """#### 🔍 Specify Data Columns"""
        mcol, vtcol = st.columns(2)

        with mcol:
            st.session_state["metric_column"] = metric_column = st.selectbox(
                "Metric Column",
                [st.session_state.get("metric_column", None)]
                + st.session_state["available_data_columns"],
                help="Select the column in your dataset that "
                "defines the target metric that you'd like "
                "to compare across treatment groups.",
            )

        inferred_variable_type = "continuous"
        if st.session_state.get("metric_column") is not None:
            # Filter out any invalid metric rows
            valid_metric_mask = st.session_state["data"][metric_column].notnull()
            n_invalid_metric = (~valid_metric_mask).sum()
            if n_invalid_metric > 0:
                st.session_state["data"] = st.session_state["data"][valid_metric_mask]
                st.warning(
                    f"Removing {n_invalid_metric:,} "
                    f"observations due to invalid values in `{metric_column}` column"
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
                help="""Select the variable type of your data.

For conversion metrics,
encoded as True/False or 0/1 (e.g. conversion on a CTA), use the
`binary` metric type. For event count metrics (e.g. # of clicks in a session),
use the `counts` variable. For continuous variables (e.g. session length
in seconds or proportion of time spent on a page) use the `continuous`
variable type.

> ⚠️ Note that the app will try to infer the variable type from the values in the
`Metric Column`. This option can be used to override that that inferred variable type
                """,
            )
            st.session_state["variable_type"] = variable_type

        tcol, _ = st.columns(2)
        with tcol:
            st.session_state["treatment_column"] = treatment_column = st.selectbox(
                "Treatment Column",
                [st.session_state.get("treatment_column", None)]
                + st.session_state["available_data_columns"],
                help="Select the column in your dataset that "
                "defines the individual treatment groups to compare. "
                "Note that this column should contain discrete values.",
            )

        if treatment_column is not None:
            # Filter out any invalid treatment rows
            valid_treatment_mask = st.session_state["data"][treatment_column].notnull()

            n_invalid_treatments = (~valid_treatment_mask).sum()
            if n_invalid_treatments > 0:
                st.session_state["data"] = st.session_state["data"][
                    valid_treatment_mask
                ]
                st.warning(
                    f"Removing {n_invalid_treatments:,} "
                    f"observations due to invalid values in `{treatment_column}` column"
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
                    help="Select the value in `Treatment Column` that specifies "
                    "the control group for the experiment.",
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
                    help="Select one or more values in `Treatment Column` that specify "
                    "the variation groups to compare to the control group.",
                )
            st.session_state["n_variations"] = len(variation_groups)

            control_samples = get_samples(
                st.session_state.data,
                st.session_state.metric_column,
                st.session_state.treatment_column,
                st.session_state.control_group,
            )
            variation_samples = []
            for treatment_name in st.session_state["variation_groups"]:
                variation_samples.append(
                    get_samples(
                        st.session_state.data,
                        st.session_state.metric_column,
                        st.session_state.treatment_column,
                        treatment_name,
                    )
                )

            """### Data Summary"""
            summarize_samples([control_samples] + variation_samples)

            st.write(plot_samples(control_samples, variation_samples, variable_type))

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
                help="Select the relevant hypothesis comparison type "
                "for your experiment",
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
            help="Set [the acceptable False positive rate](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors) for your Hypothesis "
            "test. For example, alpha=0.05 means that we're willing to accept a "
            "false positive (i.e. we detect a difference between the variation "
            " and control when there isn't one) in one out of twenty experiments.",
        )

    # -------- Inference Specification -----------

    """## 🛠️ Inference Procedure"""

    icol, ocol = st.columns(2)

    INFERENCE_METHODS = ["frequentist", "bootstrap", "bayesian"]

    if (st.session_state.get("metric_column") is not None) and (
        st.session_state.get("treatment_column") is not None
    ):
        max_treatment_nobs = (
            st.session_state["data"]
            .groupby(st.session_state["treatment_column"])
            .count()[st.session_state["metric_column"]]
            .max()
        )

        if max_treatment_nobs > MAX_BOOTSTRAP_NOBS:
            st.warning(
                f"Dataset contains at least {max_treatment_nobs:,} observations "
                "for one of the treatments. Bootstrap inference is not available "
                "to ensure computational stability."
            )
            INFERENCE_METHODS.remove("bootstrap")

    with icol:
        st.session_state["inference_method"] = inference_method = st.selectbox(
            label="""Inference Method
            """,
            options=INFERENCE_METHODS,
            index=INFERENCE_METHODS.index(
                st.session_state.get("inference_method", "frequentist")
            ),
            help="""Select the inference method used for the Hypothesis test.
            Different inference methods have varying trade-offs in terms
            of interpretation, numerical efficiency, etc. Furthermore,
            different methods will have different options that will pop up on
            the right of this dropdown. For details, see the docs on [Frequentist Inference](https://en.wikipedia.org/wiki/Frequentist_inference),
            [Bootstrap Inference](https://en.wikipedia.org/wiki/Bootstrapping_\(statistics\)), and
            [Bayesian Inference](https://en.wikipedia.org/wiki/Bayesian_inference).
            If you're unsure about the inference method to use, or how to set 
            its options, you can always just stick
            with the default method 😉.""",
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
                help="""Define a custom statistic function for the bootstrap.
                You can use numpy (np) or scipy (sp) functions in the
                definition. The function must return a scalar statistic value when
                given an array of numbers. For example `lambda x: sp.linalg.norm(np.abs(x))`
                would be a valid statistic function.
                """,
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
                help="""Select the model form used for Bayesian inference. If you're 
                unsure about which model to use, or how to configure, feel free to 
                use the default model name 😉.""",
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
                help="Select the parameter estimation method for Bayesian inference. "
                "If you're unsure about which estimation method to use, feel free to "
                "use the defaults, which should generally provide good results 🤓.",
            )
            inference_params[
                "bayesian_parameter_estimation_method"
            ] = bayesian_parameter_estimation_method
        st.session_state["inference_params"] = inference_params

    # -------- Analysis -----------

    """## ⚡️ Analysis"""

    run_analysis = st.button(
        label="Run Analysis",
        help="Run the inference procedure on your dataset. This will generate "
        "a results report below.",
    )
    if run_analysis:
        run_inference()

        """#### 📊 Test Results"""
        plot_results()

        rcol1, rcol2 = st.columns([0.3, 0.7])

        with rcol1:
            """#### Test Summary"""
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
