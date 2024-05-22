import datasets
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar


class BASE:
    def __init__(
        self,
        train_datas: datasets.Dataset,
        key_chosen: str,
        key_rejected: str,
        key_filped: str,
        epsilon: float,
        valid_datas: datasets.Dataset = None,
    ) -> None:
        self.train_datas = train_datas
        if key_filped is not None:
            self.original_train_values = [
                train_datas[i][key_chosen] - train_datas[i][key_rejected]
                for i in range(train_datas.num_rows)
                if train_datas[i][key_filped] == 0
            ]
            self.adversarial_train_values = [
                train_datas[i][key_chosen] - train_datas[i][key_rejected]
                for i in range(train_datas.num_rows)
                if train_datas[i][key_filped] == 1
            ]
            self.train_values = (
                self.original_train_values + self.adversarial_train_values
            )
        else:
            self.train_values = [
                train_datas[i][key_chosen] - train_datas[i][key_rejected]
                for i in range(train_datas.num_rows)
            ]
            self.original_train_values = None
            self.adversarial_train_values = None
        self.valid_datas = valid_datas
        if valid_datas is not None:
            if key_filped is not None:
                self.original_valid_values = [
                    valid_datas[i][key_chosen] - valid_datas[i][key_rejected]
                    for i in range(valid_datas.num_rows)
                    if valid_datas[i][key_filped] == 0
                ]
                self.adversarial_valid_values = [
                    valid_datas[i][key_chosen] - valid_datas[i][key_rejected]
                    for i in range(valid_datas.num_rows)
                    if valid_datas[i][key_filped] == 1
                ]
                self.valid_values = (
                    self.original_valid_values + self.adversarial_valid_values
                )
            else:
                self.valid_values = [
                    valid_datas[i][key_chosen] - valid_datas[i][key_rejected]
                    for i in range(valid_datas.num_rows)
                ]
                self.original_valid_values = None
                self.adversarial_valid_values = None
        else:
            self.valid_values = None

        self.key_chosen = key_chosen
        self.key_rejected = key_rejected
        self.key_filped = key_filped
        self.threshold, self.detach = self.calc_threshold()

    def __call__(self, sample) -> bool:
        return sample[self.key_chosen] - sample[self.key_rejected] < self.threshold

    def calc_threshold(self) -> float:
        return 0, 0.5

    def get_detach(self):
        return self.threshold, self.detach

    def get_result(self) -> float:
        sum, cnt = 0, 0
        for i in range(self.train_datas.num_rows):
            data = self.train_datas[i]
            sum = sum + 1
            cnt = cnt + (self.__call__(data) == data[self.key_filped])
        return cnt / sum

    def push_to_hub(self, save_name, token=None):
        dataset = {"train": self.train_datas}
        if self.valid_datas is not None:
            dataset["valid"] = self.valid_datas
        dataset = datasets.DatasetDict(dataset)
        if token is not None:
            dataset.push_to_hub(save_name, private=True)
        else:
            dataset.push_to_hub(save_name, private=True, token=token)

    def calc_bound(self, size):
        left_size = size // 2
        right_size = size - left_size
        self.train_values.sort()
        return self.train_values[left_size + 1], self.train_values[-right_size - 1]


class GMM(BASE):
    def __init__(
        self,
        train_datas: datasets.Dataset,
        key_chosen: str,
        key_rejected: str,
        key_filped: str,
    ) -> None:
        self.mu_fit, self.sigma_fit, self.alpha_fit = None, None, None
        super().__init__(train_datas, key_chosen, key_rejected, key_filped, 0)

    def weighted_area(self, x, mu, sigma, alpha):
        cdf_a = norm.cdf(x, loc=mu, scale=sigma)
        cdf_b = norm.cdf(x, loc=-mu, scale=sigma)
        area_a = cdf_a
        area_b = 1 - cdf_b
        return -1 * (area_a * alpha + area_b * (1 - alpha))

    def bimodal(self, x, mu, sigma, alpha):
        return alpha * norm.pdf(x, mu, sigma) + (1 - alpha) * norm.pdf(x, -mu, sigma)

    def calc_threshold(self) -> float:
        values = self.original_train_values + self.adversarial_train_values
        hist, bin_edges = np.histogram(values, bins=100, density=True)
        max_peak_index = np.argmax(hist)
        max_peak_value = max(min(hist[max_peak_index], 0.5), 0)
        mu_guess = max(
            min((bin_edges[max_peak_index] + bin_edges[max_peak_index + 1]) / 2, -0.25),
            -5,
        )
        sigma_guess = 1
        alpha_guess = max_peak_value
        initial_guess = [
            mu_guess,
            sigma_guess,
            alpha_guess,
        ]
        popt, pcov = curve_fit(
            self.bimodal,
            bin_edges[:-1],
            hist,
            p0=initial_guess,
            bounds=([-np.inf, 0, 0], [0, np.inf, 1]),
        )
        mu_fit, sigma_fit, alpha_fit = popt
        self.mu_fit, self.sigma_fit, self.alpha_fit = mu_fit, sigma_fit, alpha_fit
        result = minimize_scalar(
            self.weighted_area,
            args=(mu_fit, sigma_fit, alpha_fit),
            bounds=(mu_fit, -mu_fit),
            method="bounded",
        )
        return result.x, -result.fun

    def calc_bound(self, size):
        left_size = round(size * min(self.alpha_fit, 1))
        right_size = size - left_size
        self.train_values.sort()
        return self.train_values[left_size + 1], self.train_values[-right_size - 1]
