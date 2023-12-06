import warnings
import logging

import numpy as np
import pandas as pd

import holoviews as hv
import hvplot.pandas

from functools import reduce
from holoviews.plotting.mpl import MPLPlot


from spearmint import vis

warnings.filterwarnings("ignore", module="holoviews")
MPLPlot.sublabel_format = ""


SHOW_INTERVAL_TEXT = False
HISTOGRAM_ALPHA = 0.3


CONTROL_COLOR = vis.COLORS.gray
variation_colors = [
    "blue",
    "green",
    "red",
    "yellow",
    "purple",
    "light_blue",
    "light_green",
    "salmon",
]
n_colors = len(variation_colors)


def get_variation_color(idx):
    return getattr(vis.COLORS, variation_colors[idx % n_colors])


def rescale_color(hex_color_str: str, scale_factor: float = 1.0):
    """
    Rescales a hex color string by `scale_factor`. Returns scaled hex string.

    Parameters
    ----------
    hex_color_str : str
        The original color string in HEX
    scale_factor : float
        The color scale factor. To darken the color, use a float value between
        0 and 1. To brighten the color, use a float value greater than 1.

    >>> rescale_color("#DF3C3C", .5)
    #6F1E1E
    >>> rescale_color("#52D24F", 1.6)
    #83FF7E
    >>> rescale_color("#4F75D2", 1)
    #4F75D2
    """

    def _clamp(val, minimum=0, maximum=255):
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return int(val)

    hex_color_str = hex_color_str.strip("#")

    if scale_factor < 0 or len(hex_color_str) != 6:
        return hex_color_str

    r, g, b = (
        int(hex_color_str[:2], 16),
        int(hex_color_str[2:4], 16),
        int(hex_color_str[4:], 16),
    )

    r = _clamp(r * scale_factor)
    g = _clamp(g * scale_factor)
    b = _clamp(b * scale_factor)

    return "#%02x%02x%02x" % (r, g, b)


def plot_samples_kde(control_samples, variation_samples, variable_type):
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
    return layout


def plot_binary_samples_bars(control_samples, variation_samples):
    def _make_dataframe(samples, is_control: bool = False):
        unique_values = np.unique(samples.data).astype(int)
        name = samples.name
        if is_control:
            name += " (control)"
        nobs = samples.nobs
        treatments = []
        values = []
        proportions = []
        for uv in unique_values:
            n_values = (samples.data == uv).sum()
            treatments.append(name)
            values.append(uv)
            proportions.append(n_values / nobs)

        return pd.DataFrame(
            {"treatment": treatments, "value": values, "proportion": proportions}
        )

    dfs = [_make_dataframe(control_samples, is_control=True)]
    colors = [CONTROL_COLOR]
    for ii, vs in enumerate(variation_samples):
        dfs.append(_make_dataframe(vs))
        colors.append(get_variation_color(ii))

    df = pd.concat(dfs, axis=0).reset_index(drop=True)

    layout = df.hvplot.bar(
        x="treatment",
        y="proportion",
        by="value",
        # color="color",
        # stacked=True,
    ).opts(multi_level=True, invert_axes=True, color=colors, alpha=0.75)
    return layout


def plot_samples_histogram(control_samples, variation_samples, variable_type):
    max_vals = [control_samples.data.max()] + [
        vs.data.max() for vs in variation_samples
    ]
    max_val = np.max(max_vals)

    min_vals = [control_samples.data.min()] + [
        vs.data.min() for vs in variation_samples
    ]
    min_val = np.min(min_vals)

    if variable_type == "binary":
        # bins = np.linspace(0, 1, 20)
        return plot_binary_samples_bars(control_samples, variation_samples)

    elif variable_type == "counts":
        bins = np.arange(0, max_val + 1)
    else:
        bins = np.linspace(min_val, max_val, 20)

    layout = vis.plot_histogram(
        samples=control_samples,
        label=f"{control_samples.name} (control)",
        color=CONTROL_COLOR,
        bins=bins,
        alpha=HISTOGRAM_ALPHA,
    )
    for ii, vs in enumerate(variation_samples):
        layout *= vis.plot_histogram(
            samples=vs,
            label=vs.name,
            color=get_variation_color(ii),
            bins=bins,
            alpha=HISTOGRAM_ALPHA,
        )
    return layout


def plot_samples(control_samples, variation_samples, variable_type):
    nobs = [control_samples.nobs] + [vs.nobs for vs in variation_samples]
    if np.max(nobs) > 500 and variable_type == "continuous":
        try:
            layout = plot_samples_kde(control_samples, variation_samples, variable_type)
        except Exception as e:
            logging.warning(
                f"Could not plot kde, falling back to histogram. Exception: `{e}`"
            )
            layout = plot_samples_histogram(
                control_samples, variation_samples, variable_type
            )
    else:
        layout = plot_samples_histogram(
            control_samples, variation_samples, variable_type
        )

    layout = layout.relabel("Treatment Samples").opts(legend_position="right")

    return hv.render(layout)


def plot_frequentist_continuous_results(results):
    distribution_plots = []
    # Sample distributinos
    distribution_plots.append(
        vis.plot_gaussian(
            mean=results[0].control.mean,
            std=results[0].control.std,
            label=results[0].control.name + " (control)",
            color=CONTROL_COLOR,
        )
    )

    distribution_plots.append(
        vis.plot_interval(
            *results[0].control.confidence_interval(1 - results[0].alpha),
            middle=results[0].control.mean,
            label=results[0].control.name + " (control)",
            color=CONTROL_COLOR,
            show_interval_text=SHOW_INTERVAL_TEXT,
        )
    )

    for ii, result in enumerate(results):
        # Variation components
        distribution_plots.append(
            vis.plot_gaussian(
                mean=result.variation.mean,
                std=result.variation.std,
                label=result.variation.name,
                color=get_variation_color(ii),
            )
        )

        distribution_plots.append(
            vis.plot_interval(
                *result.variation.confidence_interval(1 - result.alpha),
                middle=result.variation.mean,
                label=result.variation.name,
                color=get_variation_color(ii),
                show_interval_text=SHOW_INTERVAL_TEXT,
            )
        )

    distribution_plot = reduce(lambda x, y: x * y, distribution_plots)
    distribution_plot = distribution_plot.relabel(
        "Sample Distribution and\nCentral Tendency Estimates"
    ).opts(legend_position="left", xlabel="Value", ylabel="pdf")

    # Delta distribution
    delta_plots = []
    max_pdf_height = 0
    for ii, result in enumerate(results):
        mean_delta = result.variation.mean - result.control.mean
        std_delta = (
            (result.control.var / result.control.nobs)
            + (result.variation.var / result.control.nobs)
        ) ** 0.5

        delta_dist = vis.plot_gaussian(
            mean=mean_delta,
            std=std_delta,
            label=f"{result.variation.name}",
            color=get_variation_color(ii),
        )

        left_bound = result.delta_confidence_interval[0]
        right_bound = result.delta_confidence_interval[1]

        max_pdf_height = np.max((max_pdf_height, delta_dist.data["pdf"].max()))

        delta_ci = vis.plot_interval(
            left_bound,
            right_bound,
            mean_delta,
            color=get_variation_color(ii),
            label="",
            show_interval_text=SHOW_INTERVAL_TEXT,
            vertical_offset=-(max_pdf_height * 0.01),
        )
        delta_plots.append(delta_dist)
        delta_plots.append(delta_ci)

    zero_delta_vline = hv.Curve(
        ([0.0, 0.0], [0.0, max_pdf_height]), vdims="pdf", label="Null Delta"
    ).opts(color=CONTROL_COLOR)

    delta_plots.append(zero_delta_vline)

    delta_plot = reduce(lambda x, y: x * y, delta_plots)
    delta_plot = (
        delta_plot.relabel("Means Delta")
        .opts(xlabel="delta", ylabel="pdf")
        .opts(legend_position="right")
    )

    visualization = distribution_plot + delta_plot
    visualization.opts(shared_axes=False)

    return visualization


def plot_frequentist_binary_results(results):
    # Sample distribution comparison plot
    def get_binomial_cis(samples, alpha):
        """Convert proportionality to successful # trials"""
        confidence = 1 - alpha
        cis = np.round(
            np.array(samples.confidence_interval(confidence=confidence)) * samples.nobs
        ).astype(int)

        mean = np.round(samples.mean * samples.nobs).astype(int)
        return cis[0], cis[1], mean

    distribution_plots = []
    distribution_plots.append(
        vis.plot_binomial(
            p=results[0].control.mean,
            n=results[0].control.nobs,
            label=f"{results[0].control.name} (control)",
            color=CONTROL_COLOR,
        ).opts(axiswise=True)
    )
    distribution_plots.append(
        vis.plot_interval(
            *get_binomial_cis(results[0].control, results[0].alpha),
            label=f"{results[0].control.name} (control)",
            color=CONTROL_COLOR,
            show_interval_text=SHOW_INTERVAL_TEXT,
        )
    )

    for ii, result in enumerate(results):
        distribution_plots.append(
            vis.plot_binomial(
                p=result.variation.mean,
                n=result.variation.nobs,
                label=result.variation.name,
                color=get_variation_color(ii),
            ).opts(axiswise=False)
        )

        distribution_plots.append(
            vis.plot_interval(
                *get_binomial_cis(result.variation, result.alpha),
                label=result.variation.name,
                color=get_variation_color(ii),
                show_interval_text=SHOW_INTERVAL_TEXT,
            )
        )

    distribution_plot = reduce(lambda x, y: x * y, distribution_plots)
    distribution_plot = distribution_plot.relabel(
        "Sample Distribution and\nCentral Tendency Estimates"
    ).opts(legend_position="left", xlabel="# Successful Trials", ylabel="pdf")

    # Delta distribution plot
    delta_plots = []
    max_pdf_height = 0.0
    for ii, result in enumerate(results):
        mean_delta = result.variation.mean - result.control.mean
        std_delta = (
            (result.control.var / result.control.nobs)
            + (result.variation.var / result.control.nobs)
        ) ** 0.5

        delta_dist = vis.plot_gaussian(
            mean=mean_delta,
            std=std_delta,
            label=f"{result.variation.name}",
            color=get_variation_color(ii),
        )

        left_bound = result.delta_confidence_interval[0]
        right_bound = result.delta_confidence_interval[1]

        max_pdf_height = np.max((max_pdf_height, delta_dist.data["pdf"].max()))

        delta_ci = vis.plot_interval(
            left_bound,
            right_bound,
            mean_delta,
            color=get_variation_color(ii),
            label="",
            show_interval_text=SHOW_INTERVAL_TEXT,
            vertical_offset=-(max_pdf_height * 0.01),
        )

        delta_plots.append(delta_dist)
        delta_plots.append(delta_ci)

    zero_delta_vline = hv.Curve(
        ([0.0, 0.0], [0.0, max_pdf_height]), vdims="pdf", label="Null Delta"
    ).opts(color=CONTROL_COLOR)
    delta_plots.append(zero_delta_vline)

    delta_plot = reduce(lambda x, y: x * y, delta_plots)
    delta_plot = (
        delta_plot.relabel("Proportionality Delta")
        .opts(xlabel="delta", ylabel="pdf")
        .opts(legend_position="right")
    )

    visualization = distribution_plot + delta_plot
    visualization.opts(shared_axes=False)

    return visualization


def plot_frequentist_counts_resuls(results):
    distribution_plots = []
    distribution_plots.append(
        vis.plot_poisson(
            mu=results[0].control.mean,
            label=f"{results[0].control.name} (control)",
            color=CONTROL_COLOR,
        )
    )
    # Proportion/conversion rate confidence intervals plot
    distribution_plots.append(
        vis.plot_interval(
            *results[0].control.confidence_interval(1 - results[0].alpha),
            middle=results[0].control.mean,
            label=f"{results[0].control.name} (control)",
            color=CONTROL_COLOR,
            show_interval_text=SHOW_INTERVAL_TEXT,
        )
    )

    for ii, result in enumerate(results):
        distribution_plots.append(
            vis.plot_poisson(
                mu=result.variation.mean,
                label=result.variation.name,
                color=get_variation_color(ii),
            )
        )

        distribution_plots.append(
            vis.plot_interval(
                *result.variation.confidence_interval(1 - result.alpha),
                middle=result.variation.mean,
                label=result.variation.name,
                color=get_variation_color(ii),
                show_interval_text=SHOW_INTERVAL_TEXT,
            )
        )

    distribution_plot = reduce(lambda x, y: x * y, distribution_plots)
    distribution_plot = distribution_plot.relabel(
        "Sample Distribution and\nCentral Tendency Estimates"
    ).opts(legend_position="left", xlabel="N Events", ylabel="pdf")

    delta_plots = []
    max_pdf_height = 0
    for ii, result in enumerate(results):
        # Delta distribution plot
        mean_ratio = result.variation.mean / result.control.mean
        std_delta = (
            (result.control.var / result.control.nobs)
            + (result.variation.var / result.control.nobs)
        ) ** 0.5

        delta_dist = vis.plot_gaussian(
            mean=mean_ratio,
            std=std_delta,
            label=f"{result.variation.name}",
            color=get_variation_color(ii),
        )

        left_bound = result.delta_confidence_interval[0]
        right_bound = result.delta_confidence_interval[1]

        max_pdf_height = np.max((max_pdf_height, delta_dist.data["pdf"].max()))

        delta_ci = vis.plot_interval(
            left=left_bound,
            right=right_bound,
            middle=mean_ratio,
            color=get_variation_color(ii),
            label="",
            show_interval_text=SHOW_INTERVAL_TEXT,
            vertical_offset=-(max_pdf_height * 0.01),
        )

        delta_plots.append(delta_dist)
        delta_plots.append(delta_ci)

    one_delta_vline = hv.Curve(
        ([1.0, 1.0], [0.0, max_pdf_height]), vdims="pdf", label="Null Ratio"
    ).opts(color=CONTROL_COLOR)
    delta_plots.append(one_delta_vline)

    delta_plot = reduce(lambda x, y: x * y, delta_plots)
    delta_plot = (
        delta_plot.relabel("Rates Ratio")
        .opts(xlabel="ratio", ylabel="pdf")
        .opts(legend_position="right")
    )

    visualization = distribution_plot + delta_plot
    visualization.opts(shared_axes=False)

    return visualization


def plot_bootstrap_results(results):
    test_statistic_label = results[0].test_statistic_name.replace("_", " ")
    test_statistic_title = test_statistic_label.title()

    _alpha = results[0].alpha

    ci_bounds = np.round(100 * np.array((_alpha / 2, 1 - _alpha / 2)))

    control_samples = results[0].aux["control_bootstrap_samples"]
    control_name = f"{results[0].control.name} (control)"
    distribution_plots = []
    delta_plots = []
    distribution_plots.append(
        vis.plot_kde(
            samples=control_samples.data,
            label=f"{control_name}",
            color=CONTROL_COLOR,
        )
    )

    # Confidence intervals
    distribution_plots.append(
        vis.plot_interval(
            *control_samples.percentiles(ci_bounds),
            middle=control_samples.mean,
            label=f"{control_name}",
            color=CONTROL_COLOR,
            show_interval_text=SHOW_INTERVAL_TEXT,
        )
    )

    max_pdf_height = 0
    for ii, result in enumerate(results):
        variation_samples = result.aux["variation_bootstrap_samples"]

        distribution_plots.append(
            vis.plot_kde(
                samples=variation_samples.data,
                label=result.variation.name,
                color=get_variation_color(ii),
            )
        )

        distribution_plots.append(
            vis.plot_interval(
                *variation_samples.percentiles(ci_bounds),
                middle=variation_samples.mean,
                label=result.variation.name,
                color=get_variation_color(ii),
                show_interval_text=SHOW_INTERVAL_TEXT,
            )
        )

        delta_samples = result.aux["delta_bootstrap_samples"]
        delta_dist = vis.plot_kde(
            samples=delta_samples.data,
            label=f"{result.variation.name}",
            color=get_variation_color(ii),
        )

        max_pdf_height = np.max((max_pdf_height, delta_dist.data["pdf"].max()))
        mean_delta = result.aux["delta_bootstrap_samples"].mean
        delta_ci = vis.plot_interval(
            *delta_samples.percentiles(ci_bounds),
            mean_delta,
            color=get_variation_color(ii),
            label="",
            show_interval_text=SHOW_INTERVAL_TEXT,
            vertical_offset=-(max_pdf_height * 0.01),
        )
        delta_plots.append(delta_dist)
        delta_plots.append(delta_ci)

    zero_delta_vline = hv.Curve(
        ([0.0, 0.0], [0.0, max_pdf_height]), vdims="pdf", label="Null"
    ).opts(color=CONTROL_COLOR)

    delta_plots.append(zero_delta_vline)

    distribution_plot = reduce(lambda x, y: x * y, distribution_plots)
    distribution_plot = distribution_plot.relabel(
        f"{test_statistic_title} Comparison"
    ).opts(legend_position="left", xlabel=test_statistic_label, ylabel="pdf")

    delta_plot = reduce(lambda x, y: x * y, delta_plots)
    delta_plot_title = f"{test_statistic_title} Delta"
    delta_plot = (
        delta_plot.relabel(delta_plot_title)
        .opts(xlabel="delta", ylabel="pdf")
        .opts(legend_position="right")
    )

    visualization = distribution_plot + delta_plot
    visualization.opts(shared_axes=False)

    return visualization


def plot_bayesian_results(
    results,
    include_prior: bool = False,
):
    from spearmint.inference.bayesian.bayesian_inference import _get_delta_param

    control_name = f"{results[0].control.name} (control)"
    control_posterior = results[0].control_posterior
    credible_mass = 1 - results[0].alpha
    comparison_param = _get_delta_param(results[0].model_name)

    delta_plots = []
    distribution_plots = []
    distribution_plots.append(
        vis.plot_kde(
            samples=control_posterior.data,
            label=control_name,
            color=CONTROL_COLOR,
        )
    )

    # Confidence intervals
    distribution_plots.append(
        vis.plot_interval(
            *control_posterior.hdi(credible_mass),
            middle=control_posterior.mean,
            label=control_name,
            color=CONTROL_COLOR,
            show_interval_text=SHOW_INTERVAL_TEXT,
        )
    )

    max_pdf_height = 0.0
    for ii, result in enumerate(results):
        delta_samples = result.delta_posterior
        variation_name = result.variation.name
        variation_color = get_variation_color(ii)
        variation_posterior = result.variation_posterior

        distribution_plots.append(
            vis.plot_kde(
                samples=variation_posterior.data,
                label=variation_name,
                color=variation_color,
            )
        )

        distribution_plots.append(
            vis.plot_interval(
                *variation_posterior.hdi(credible_mass),
                middle=variation_posterior.mean,
                label=variation_name,
                color=variation_color,
                show_interval_text=SHOW_INTERVAL_TEXT,
            )
        )

        delta_dist = vis.plot_kde(
            samples=delta_samples.data,
            label=f"{variation_name}",
            color=variation_color,
        )

        max_pdf_height = np.max((max_pdf_height, delta_dist.data["pdf"].max()))
        mean_delta = delta_samples.mean

        delta_ci = vis.plot_interval(
            *delta_samples.hdi(credible_mass),
            mean_delta,
            color=variation_color,
            label="",
            show_interval_text=SHOW_INTERVAL_TEXT,
            vertical_offset=-(max_pdf_height * 0.01),
        )
        delta_plots.append(delta_dist)
        delta_plots.append(delta_ci)

    vline = hv.Curve(
        ([0.0, 0.0], [0.0, max_pdf_height]), vdims="pdf", label="NULL"
    ).opts(color=CONTROL_COLOR)
    delta_plots.append(vline)

    if include_prior:
        distribution_plots.append(
            vis.plot_kde(
                samples=results[0].prior.data,
                label="prior",
                color=vis.COLORS.light_gray,
            )
        )

    distribution_plot = reduce(lambda x, y: x * y, distribution_plots)
    distribution_plot = distribution_plot.relabel(
        f"Posterior {comparison_param} Comparison"
    ).opts(
        legend_position="left",
        xlabel=f"Posterior {comparison_param}",
        ylabel="pdf",
    )

    delta_plot = reduce(lambda x, y: x * y, delta_plots)
    delta_plot_title = f"Posterior {comparison_param} Delta"
    delta_plot = (
        delta_plot.relabel(delta_plot_title)
        .opts(xlabel="delta", ylabel="pdf")
        .opts(legend_position="right")
    )

    visualization = distribution_plot + delta_plot
    visualization.opts(shared_axes=False)

    return visualization
