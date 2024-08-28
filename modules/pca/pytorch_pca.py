from typing import Optional

import torch
import plotly.graph_objects as go


class PrincipalComponentAnalysis:
    """
    Implements Principal Component Analysis (PCA) using PyTorch by leveraging
    Singular Value Decomposition (SVD).
    """
    _validation_rules = {
        "num_components": int
    }

    def __init__(
            self,
            num_components: Optional[int] = None) -> None:
        """
        Initializes the PCA instance with a device and number of components.

        :param Optional[int] num_components: Number of principal components\
            to retain. If not provided, it will be determined based on the\
            data during fitting. Default is None.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.num_components = num_components

        self.eigen_vectors = None
        self.original_ratios = None
        self.variance_ratios = None
        self._variances = None

    def fit(
            self,
            data: torch.Tensor,
            compute_full: bool = False,
            desired_ratio: float = 0.98,
            variance_margin: float = 0.02) -> None:
        """
        Fit the PCA model to the provided dataset using SVD.

        :param torch.Tensor data: The dataset to fit the model to,\
            expected shape (n_samples, n_features).
        :param bool compute_full: Whether to compute the full matrix\
            decomposition. Default is False.
        :param float desired_ratio: Target ratio for explained variance.\
            Default is 0.98.
        :param float variance_margin: Margin for the acceptable ratio of\
            explained variance. Default is 0.02.
        """
        if self.num_components is None:
            self.num_components = data.shape[1]
            self._perform_decomposition(data, compute_full)
            self.original_ratios = self.variance_ratios.clone()
            self.num_components = self._determine_optimal_components(
                self.original_ratios, desired_ratio, variance_margin)
            self._perform_decomposition(data, compute_full)
        else:
            self._perform_decomposition(data, compute_full)

        self._variances = self._variances[:self.num_components]
        self.variance_ratios = self.variance_ratios[:self.num_components]

        explained_percentage = torch.cumsum(
            self.variance_ratios, dim=0)[-1].item() * 100

        exp_info = f"Explained {explained_percentage:.2f}% of the variance"
        exp_info += f" with {self.num_components} components.\n"
        exp_info += f"PrincipalComponentAnalysis("
        exp_info += f"num_components={self.num_components})"
        print(exp_info)

    def _perform_decomposition(
            self,
            data: torch.Tensor,
            compute_full: bool = False) -> None:
        """
        Perform the Singular Value Decomposition on the provided data.

        :param torch.Tensor data: Data on which SVD is to be performed.
        :param bool compute_full: Whether to compute the full SVD or\
            reduced form. Default is False.
        """
        data = data.to(self.device).float()
        data = data - data.mean(dim=0)

        _, singular_values, eigen_vectors = torch.linalg.svd(
            data, full_matrices=compute_full)

        self.eigen_vectors = eigen_vectors[:, :self.num_components]

        self._variances = singular_values[:self.num_components] ** 2 / (
            data.shape[0] - 1)
        total_variance = torch.sum(singular_values ** 2) / (
            data.shape[0] - 1)

        self.variance_ratios = self._variances / total_variance

    def _determine_optimal_components(
            self,
            ratios: torch.Tensor,
            target_ratio: float = 0.98,
            margin: float = 0.02) -> int:
        """
        Determine the optimal number of components, based on the desired\
            explained variance ratio.

        :param torch.Tensor ratios: Cumulative explained variance ratios.
        :param float target_ratio: Target ratio for explained variance.\
            Default is 0.98.
        :param float margin: Acceptable margin variance from the target ratio.\
            Default is 0.02.

        :return int: Optimal number of components.
        """
        cumulative_ratios = torch.cumsum(ratios, dim=0)
        lower_bound = target_ratio - margin
        upper_bound = target_ratio + margin

        valid_indices = torch.where((cumulative_ratios >= lower_bound) & (
            cumulative_ratios <= upper_bound))[0]

        if len(valid_indices) == 0:
            err = "No suitable number of components found "
            err += "within the specified ratio bounds."
            raise ValueError(err)

        return valid_indices[0].item() + 1

    def transform(
            self,
            data: torch.Tensor) -> torch.Tensor:
        """
        Apply the PCA transformation, using the fitted model.

        :param torch.Tensor data: Data to transform of shape\
            (number_of_samples, number_of_features).

        :return torch.Tensor: Transformed data of shape\
            (number_of_samples, num_components).
        """
        if self.eigen_vectors is None:
            raise ValueError("Model must be fitted before transformation.")

        data = data.to(self.device).float()
        centered_data = data - data.mean(dim=0)

        return torch.matmul(centered_data, self.eigen_vectors).cpu()

    def fit_transform(
            self,
            data: torch.Tensor,
            compute_full: bool = False,
            desired_ratio: float = 0.98,
            variance_margin: float = 0.02) -> torch.Tensor:
        """
        Fit the PCA model and transform the data in one step.

        :param torch.Tensor data: Data to transform of shape\
            (number_of_samples, number_of_features).
        :param bool compute_full: Whether to compute full SVD matrices.\
            Default is False.
        :param float desired_ratio: Desired explained variance ratio.\
            Default is 0.98.
        :param float variance_margin: Margin for explained variance ratio.\
            Default is 0.02.

        :return torch.Tensor: Transformed data of shape\
            (number_of_samples, num_components).
        """
        self.fit(data, compute_full, desired_ratio, variance_margin)

        return self.transform(data)

    def visualize_variance(
            self,
            plot_width: int = 850,
            plot_height: int = 550) -> None:
        """
        Visualize the explained variance as a function
        of the number of components.

        :param int plot_width: Width of the plot in pixels. Default is 850.
        :param int plot_height: Height of the plot in pixels. Default is 550.
        """
        _title = "Principal Components Explained Variance"

        if self.original_ratios is not None:
            ratio = self.original_ratios.cpu()
            marker_trace_title = "optimal num_components"
        else:
            ratio = self.variance_ratios.cpu()
            marker_trace_title = "given num_components"

        optimal_value = torch.cumsum(torch.tensor(ratio), dim=0)[
            self.num_components - 1].item()

        marker_trace = go.Scatter(
            x=[self.num_components],
            y=[optimal_value],
            mode="markers",
            marker=dict(size=8, color="red"),
            name=marker_trace_title
        )

        trace = go.Scatter(
            x=torch.arange(1, len(ratio) + 1),
            y=torch.cumsum(ratio, dim=0),
            mode="lines+markers",
            name="variance"
        )

        layout = go.Layout(
            title=dict(text=_title, x=0.5),
            xaxis=dict(
                title="Principal Components",
                tickangle=-45,
                tickvals=torch.arange(
                    0, len(ratio), 1 if len(ratio) < 20 else 2)),
            yaxis=dict(title="Explained Variance Ratio"),
            width=plot_width,
            height=plot_height
        )
        fig = go.Figure(data=[trace, marker_trace], layout=layout)
        fig.show()
