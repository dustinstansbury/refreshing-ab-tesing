"""Instructions and tooltips"""


intro = """
# Easy AB Testing

_Powered by [spearmint](https://github.com/dustinstansbury/spearmint)_

Use this simple app to run AB tests using your own dataset. The app
supports many different inference methods and variable types.
"""


class instructions:
    how_to_run_test = """
    To run a test, you'll need to:

    1. 📁 **Import a Dataset** csv file from you computer.
    2. 🔍 **Specify the data columns** that define the **metric** and **treatment groups** in your study.
    3. 💡**Specify your Hypothesis**, including comparison type and acceptable alpha. Or, just use the defaults.
    4. 🛠️ **Configure the Inference Procedure**. You can use **frequentist**, **bootstrap**, or **Bayesian** inference methods. Or, just use the defaults.
    5. ⚡️ **Run the Analysis** and interpret the reported results
    """

    interpret_test_summary = """
    ##### Top Row - Test Summary
    
    > The row of metrics on the top row give a high-level summary of the test results.
    The far left metric shows the mean for the control group. The remaining metrics
    show the mean for the variations. Below each is the relative delta--reported in percent--
    when compared to the control.
    > 
    > Variation group names with a "🟢" or "🔴" next to
    them indicate a statistically significant difference from the control. For one-tailed
    tests (`"larger"` and `"smaller"`), "🟢" indicates a significant effect in the
    direction of the hypothesis (e.g. a significant decrease when the hypothesis
    is `"smaller"`). For two-tailed tests "🔴" indicates a significant decrease, 
    while "🟢" indicates a significant increase.
    """

    interpret_frequentist_results_template = """
    ---
    
    ### Interpreting Frequentist Test Results
    
    {interpret_test_summary}
    
    
    ##### Left Plot - Sample Distribution & Central Tendency Estimates
    
    > The plot on the left shows the estimated parametric distribution that best
    describes each of the treatments, including the control (plotted in gray).
    For this test, which models {variable_type} variables, the parametric
    distribution used is the {parametric_distribution} distribution, with central
    tendency parameter `{central_tendency_param}`, that defines the expected
    {expectation_description}. 
    > 
    > Under each parametric distribution is a (1-`alpha`)% confidence interval
    (CI) for the estimates of the central tendencies. Non-overlapping CIs are a
    visual indicator that two groups are statistically different from one another.
    
    ##### Right Plot -- {comparison_type} Distribution
    
    > The plot on the right shows the {comparison_type} of estimated `{central_tendency_param}`s 
    for variation and control groups. If there is no true difference
    between the variation and control groups, then we would expect to observe a
    `Null {comparison_type}` equal to {null_comparison_value} (vertical line plotted in gray).
    > 
    > Any probability mass in the {comparison_type} distributions that overlaps
    with the  `Null {comparison_type}` indicates there is no statistical difference
    between the variation and control. Distributions that are located far from the
    `Null {comparison_type}` indicate a statistically significant difference
    from the control.
    > 
    > Under each {comparison_type} distribution is a (1-`alpha`)%
    confidence interval (CI) for the central tendency of the distribution.
    Confidence intervals that do not overlap with the `Null {comparison_type}`
    provide a visual indicator that a variant is statisticaly different from the
    control group.

    > For additional details on Frequentist Inference, please refer to the 
    > [Frequentist Inference wiki page](https://en.wikipedia.org/wiki/Frequentist_inference).
    
    """

    interpret_bootstrap_results_template = """
    ---
    
    ### Interpreting Bootstrap Test Results
    
    {interpret_test_summary}
    
    ##### Left Plot - Sample Distribution & Central Tendency Estimates
    
    > The plot on the left shows [Bootstraped Distributions](https://en.wikipedia.org/wiki/Bootstrapping_\(statistics\))
    of the sample statistic defined in `Statistic Function` field above. The
    bootstrap distribution is calculated for each of the treatments, including
    the control (plotted in gray).
    > 
    > Under each bootsrap distribution is a (1-`alpha`)% confidence interval (CI)
    for the estimates of the central tendencies. Non-overlapping CIs are a visual
    indicator that two groups are statistically different from one another.

    ##### Right Plot -- {comparison_type} Distribution
    
    > The plot on the right shows the {comparison_type} of the
    the bootstrap distributions, comparing the boostraps for the variation
    groups to that of the control group. If there is no true difference between
    the variation and control groups, then we would expect to observe a `Null {comparison_type}`
    equal to {null_comparison_value}  (vertical line plotted in gray).
    > 
    > Any probability mass in the {comparison_type} distributions that overlaps
    with the `Null {comparison_type}` indicates there is no statistical difference
    between the variation and control. Distributions that are located far from the
    `Null {comparison_type}` indicate a statistically significant difference
    from the control.
    > 
    > Under each {comparison_type} distribution is a (1-`alpha`)% confidence interval
    (CI) for the central tendency of the distribution. Confidence intervals that
    do not overlap with the `Null {comparison_type}` provide a visual indicator
    that a variant is statisticaly different from the control group.
    
    > For additional details on Bootstrap Inference, please refer to the
    > [Bootstrap Inference wiki page](https://en.wikipedia.org/wiki/Bootstrapping_\(statistics\)).
    """

    interpret_bayesian_results_template = """
    ---
    
    ### Interpreting Bayesian Test Results
    
    {interpret_test_summary}

    ##### Left Plot - Sample Distribution & Central Tendency Estimates
    
    > The plot on the left shows the posterior distribution for the
    central tendency parameter `{central_tendency_param}`. The posterior is
    estimated for each of the treatments, including the control (plotted in gray)
    > 
    > For this Bayesian test, which models {variable_type} variables, the Bayesian
    model used is the hierarchical {bayesian_model_name} model. This model
    attempts to estimate from the data the distribution over `{central_tendency_param}`,
    which describes the expected {expectation_description}.
    >
    > The light gray line shows the Prior distribution based on the settings
    configured above. Weak priors will be flatter, and cover more of the
    probability space, and thus encode more uncertainty about the true parameter
    value. Strong priors will be narrower, and thus encode more prior belief
    about the true value of the parameter.
    > 
    > Under each posterior distribution is a (1-`alpha`)% highest density interval
    (HDI) indicating where most of the probability mass is located under each
    disribution. HDIs that do not overlap with the `Null {comparison_type}`
    provide a visual indicator that a variant is statisticaly different from the
    control group.
    
    ##### Right Plot -- {comparison_type} Distribution
    
    > The plot on the right shows the {comparison_type} of `{central_tendency_param}`
    posterior distributions for variation and control groups. If there is no true
    difference between the variation and control groups, then we would expect to observe a
    `Null {comparison_type}` equal to {null_comparison_value} (vertical line plotted in gray).
    > 
    > Any probability mass in the {comparison_type} distributions that overlaps with the 
    `Null {comparison_type}` indicates there is no statistical difference between
    the variation and control. Distributions that are located far from the
    `Null {comparison_type}` indicate a statistically significant difference
    from the control.
    > 
    > Under each {comparison_type} distribution is a (1-`alpha`)% highest density
    interval (HDI). HDIs that do not overlap with the `Null {comparison_type}`
    provide a visual indicator that a variant is statisticaly different from the
    control group.

    >
    > For additional details on Bayesian Inference, please refer to the 
    > [Bayesian Inference wiki page](https://en.wikipedia.org/wiki/Bayesian_inference)
    """

    @staticmethod
    def get_interpretation(inference_method, variable_type):
        if inference_method == "frequentist":
            if variable_type == "counts":
                parametric_distribution = (
                    "[Poisson](https://en.wikipedia.org/wiki/Poisson_distribution)"
                )
                central_tendency_param = "lambda"
                expectation_description = "rate of events, or--equivalently--the number of events to occur in a standard-length trial"
                comparison_type = "Ratio"
                null_comparison_value = 1

            elif variable_type == "binary":
                parametric_distribution = (
                    "[Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)"
                )
                central_tendency_param = "p"
                expectation_description = "proportionality of successes or--equivalently--the probabiltiy of success on any given trial"
                comparison_type = "Delta"
                null_comparison_value = 0

            elif variable_type == "continuous":
                parametric_distribution = (
                    "[Gaussian](https://en.wikipedia.org/wiki/Normal_distribution)"
                )
                central_tendency_param = "mu"
                expectation_description = "value on any given trial"
                comparison_type = "Delta"
                null_comparison_value = 0

            return instructions.interpret_frequentist_results_template.format(
                interpret_test_summary=instructions.interpret_test_summary,
                variable_type=variable_type,
                parametric_distribution=parametric_distribution,
                central_tendency_param=central_tendency_param,
                expectation_description=expectation_description,
                comparison_type=comparison_type,
                null_comparison_value=null_comparison_value,
            )
        elif inference_method == "bootstrap":
            comparison_type = "Delta"
            return instructions.interpret_bootstrap_results_template.format(
                interpret_test_summary=instructions.interpret_test_summary,
                comparison_type=comparison_type,
                null_comparison_value=0,
            )
        elif inference_method == "bayesian":
            if variable_type == "counts":
                bayesian_model_name = "[Gamma-Poisson](https://www.bayesrulesbook.com/chapter-5#gamma-poisson-conjugate-family)"
                central_tendency_param = "lambda"
                expectation_description = "rate of events, or--equivalently--the number of events to occur in a standard-length trial"

            elif variable_type == "binary":
                bayesian_model_name = "[Beta-Binomial or Beta-Bernouill](https://www.bayesrulesbook.com/chapter-3#ch3-bbmodel)"
                central_tendency_param = "p"
                expectation_description = "proportionality of successes or--equivalently--the probabiltiy of success on any given trial"

            elif variable_type == "continuous":
                bayesian_model_name = "[Gaussian-Gaussian or Gaussian-Student-t](https://www.bayesrulesbook.com/chapter-5#normal-normal-conjugate-family)"
                central_tendency_param = "mu"
                expectation_description = "value on any given trial"

            return instructions.interpret_bayesian_results_template.format(
                interpret_test_summary=instructions.interpret_test_summary,
                variable_type=variable_type,
                comparison_type="Delta",
                null_comparison_value=0,
                bayesian_model_name=bayesian_model_name,
                expectation_description=expectation_description,
                central_tendency_param=central_tendency_param,
            )


class tooltip:
    file_upload = """Select a dataset file. The dataset must contain at least two
        columns, with one of the columns defining the values of a `metric` used
        for comparison. the other column should define a set of discrete values
        used for defining he control and variation `treatment`s.
        """

    select_metric_column = """
        Select the column in your dataset that "defines the target metric that
        you'd like to compare across treatment groups.
        """

    select_metric_variable_type = """
        Select the variable type of your data.

        For conversion metrics,
        encoded as True/False or 0/1 (e.g. conversion on a CTA), use the
        `binary` metric type. For event count metrics (e.g. # of clicks in a session),
        use the `counts` variable. For continuous variables (e.g. session length
        in seconds or proportion of time spent on a page) use the `continuous`
        variable type.

        > ⚠️ Note that the app will try to infer the variable type from the values in the
        `Metric Column`. This option can be used to override that that inferred
        variable type
        """

    select_treatment_column = """
        Select the column in your dataset that defines the individual treatment
        groups to compare. Note that this column should contain discrete values.
        """

    select_control_group = """
        Select the value in `Treatment Column` that specifies the control group
        for the experiment.
        """

    select_variation_groups = """
        Select one or more values in `Treatment Column` that specify the variation
        groups to compare to the control group.
        """

    select_comparison_type = """
        Select the relevant hypothesis comparison type for your experiment
        """

    set_alpha = """
        Set [the acceptable False positive rate](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors)
        for your Hypothesis test.
        
        For example, alpha=0.05 means that we're willing to accept a false positive
        (i.e. we detect a difference between the variation and control when there
        isn't one) in one out of twenty experiments.
        """

    select_mc_correction = """
        When using Frequentist or Bootstrap inference for multiple
        variation groups, one must perform [Multiple Comparison correction](https://en.wikipedia.org/wiki/Multiple_comparisons_problem) to
        avoid inflated Type I error rate. Select your correction method here. 
        
        If you're unsure about which correction method to use, you can always
        stick with the default method 😉.
        """

    select_inference_method = """
        Select the inference method used for the Hypothesis test.

        Different inference methods have varying trade-offs in terms of interpretation,
        numerical efficiency, etc. Furthermore, different methods will have
        different options that will pop up on the right of this dropdown.
        
        For details, see the docs on [Frequentist Inference](https://en.wikipedia.org/wiki/Frequentist_inference),
        [Bootstrap Inference](https://en.wikipedia.org/wiki/Bootstrapping_\(statistics\)),
        and [Bayesian Inference](https://en.wikipedia.org/wiki/Bayesian_inference).

        If you're unsure about the inference method to use, or how to set  its
        options, you can always just stick with the default method 😉.
        """

    define_statistic_function = """
        Define a custom statistic function for the bootstrap.

        The function must return a scalar statistic value when given an array of
        numbers.
        
        You can use numpy (np) or scipy (sp) functions in the definition. For
        example `lambda x: sp.linalg.norm(np.abs(x))` would be a valid statistic
        function.
        """

    select_bayesian_model = """
        Select the model form used for Bayesian inference.

        If you're unsure about which model to use, feel free to use the default
        model provided 😉.
        """

    set_bayesian_prior_mean = """
        Set the mean of the prior. This is the value that we believe, before seeing
        any data, that the value would likely take. Defaults to the global
        mean of the dataset, calculated across all treatments.
        """

    set_bayesian_prior_strength = """
        Set the strength of the prior. Strong priors will be narrower around the
        Prior Mean, and encode stronger prior belief that the true value of the
        parameter should be near the prior mean. Weaker priors are less certain
        about the true parameter value, and are thus flatter.
        """

    select_bayesian_parameters_estimation_method = """
        Select the parameter estimation method for Bayesian inference.

        If you're unsure about which estimation method to use, feel free to 
        use the defaults inference method 😉.
        """

    run_analysis = """
        Run the inference procedure on your dataset. This will generate a results
        report.
        """

    show_interpretations = """
        Provide detailed overview of the results, and how to interpret them.
        """

    show_test_details = """
        Display a dataframe of various metrics and statistics returned by the
        inference procedure. The fields in the dataframe will depend on the
        inference procedure used.
        """


class warnings:
    @staticmethod
    def removing_invalid_metrics(n_invalid_metric, metric_column):
        return f"""
        Removing {n_invalid_metric:,} observations due to invalid metric values 
        in the `{metric_column}` column.
        """

    @staticmethod
    def removing_invalid_treatments(n_invalid_treatments, treatment_column):
        return f"""
        Removing {n_invalid_treatments:,} observations due to invalid  treatment
        values  in  the `{treatment_column}` column.
        """

    @staticmethod
    def too_many_variations(n_variations, max_n_variations):
        return f"""
        Too many variations specified\n\n
        {n_variations} variations specified. To ensure numerical stability of
        this publicly-shared app, a max of {max_n_variations} variations are allowed.
        
        Please check that you've selected the correct `Treatment Column`, or reduce
        the number of variations.
        """

    @staticmethod
    def too_many_treatment_nobs_for_bootstrap(n_treatment_obs, max_treatment_obs):
        return f"""
        Bootstrap inference is unavailable.
        
        Dataset contains {n_treatment_obs:,} observations for one of the treatments.
        To ensure numerical stability on this publicly-shared application, the
        max number of observations for Bootstrap inference is limited to {max_treatment_obs:,}.
        """

    @staticmethod
    def too_few_treatment_nobs_for_bootstrap(n_treatment_obs, min_treatment_obs):
        return f"""
        Bootstrap inference is unavailable.
        
        Dataset contains only {n_treatment_obs:,} observations for one of the treatments.
        The minimum number of observations for Bootstrap inference is {min_treatment_obs:,}.
        """

    @staticmethod
    def too_few_treatment_nobs_for_analytic(n_treatment_obs, min_treatment_obs):
        return f"""
        Analytic parameter estimation is unavailable.

        Dataset contains only {n_treatment_obs:,} observations for one of the treatments.
        The minimum number of observations for Analytic parameter estimation is {min_treatment_obs:,}.
        """

    @staticmethod
    def too_many_nobs_for_mcmc(max_treatment_nobs, max_mcmc_nobs):
        return f"""
        MCMC parameter estimation is unavailable.

        The dataset contains {max_treatment_nobs:,} observations for one of the
        treatments. To ensure numerical stability on this publicly-shared
        application, the max number of observations for MCMC parameter estimation
        is limited to {max_mcmc_nobs:,}.
        """

    @staticmethod
    def too_few_treatment_nobs_for_mcmc(n_treatment_obs, min_treatment_obs):
        return f"""
        MCMC parameter estimation is unavailable.

        Dataset contains only {n_treatment_obs:,} observations for one of the treatments.
        The minimum number of observations for MCMC parameter estimation is {min_treatment_obs:,}.
        """
