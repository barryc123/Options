import math

import numpy as np
from scipy.stats import norm

N = norm.cdf
N_PRIME = norm.pdf


class BlackScholes:
    def __init__(self, S, K, r, q, t, vol, isCall) -> None:
        """
        Class containing functions for calculating price of an option and option greeks.
        :param S: Price of underlying instrument
        :param K: Strike price of option
        :param r: Interest rate
        :param q: Dividend yield (assumed to be zero)
        :param t: Time to expiry (years)
        :param vol: Volatility (sigma)
        :param isCall: Bool (True for call, False for put)
        """
        self.S = S
        self.K = K
        self.r = r
        self.q = q
        self.t = t
        self.vol = vol
        self.isCall = isCall

    def calculate_d1(self):
        return (math.log(self.S / self.K)
                + ((self.r + ((self.vol ** 2) / 2)) * self.t)) / (self.vol * math.sqrt(self.t))

    def calculate_d2(self):
        return self.calculate_d1() - (self.vol * math.sqrt(self.t))

    def calculate_option_price(self):
        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        sign = 1 if self.isCall else -1

        return (sign * self.S * N(sign * d1)) \
            - (sign * self.K * math.exp(-self.r * self.t) * N(sign*d2))

    def calculate_delta(self):
        d1 = self.calculate_d1()
        x = 0 if self.isCall else -1
        return math.exp(-self.q*self.t) * (N(d1) - x)

    def calculate_gamma(self):
        d1 = self.calculate_d1()
        # If implied vol = zero, set gamma to zero
        # Otherwise would be dividing by zero
        if self.vol == 0:
            return 0
        else:
            return (math.exp(-self.q * self.t) / (self.S * self.vol * math.sqrt(self.t))) * N_PRIME(d1)

    def calculate_theta(self):
        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        sign = 1 if self.isCall else -1

        term1 = (self.S * self.vol * math.exp(-self.q * self.t) * N_PRIME(d1)) / (2 * math.sqrt(self.t))
        term2 = self.r * self.K * math.exp(-self.r * self.t) * N(sign * d2)
        term3 = self.q * self.S * math.exp(-self.q * self.t) * N(sign * d1)

        return (1/365) * (-term1 - (sign * term2) + (sign * term3))

    def calculate_rho(self):
        d2 = self.calculate_d2()
        sign = 1 if self.isCall else -1

        return sign * (1/100) * self.K * self.t * math.exp(-self.r * self.t) * N(sign * d2)

    def calculate_vega(self):
        d1 = self.calculate_d1()

        return (1/100) * self.S * math.exp(-self.q * self.t) * math.sqrt(self.t) * N_PRIME(d1)


class ImpliedVolatility:
    def __init__(self, S, K, r, q, t, est_vol, price, isCall) -> None:
        """
        Class for calculating implied volatility of an option.
        :param S: Underlying price
        :param K: Strike price of option
        :param r: Interest rate
        :param q: Dividend yield
        :param t: Time to expiry (years)
        :param est_vol: Initial estimate of volatility
        :param price: Price of option (option premium)
        :param isCall: Bool (True for call, false for put
        """
        self.S = S
        self.K = K
        self.r = r
        self.q = q
        self.t = t
        #self.est_vol = est_vol
        self.price = price
        self.isCall = isCall

    def estimate_inital_vol(self):
        """
        Manaster and Koehler method for estimating initial volatility
        :return: Initial estimate of volatility
        """
        initial_vol_est = math.sqrt(2 * abs(math.log(self.S / self.K) / self.t + self.r))
        initial_vol_est = max(0.01, initial_vol_est)

        return initial_vol_est

    def newton_raphson(self):
        """
        Use Newton-Raphson method to calculate Implied Vol
        :return: Implied Volatility
        """
        tolerance = 1e-8
        implied_vol = self.estimate_inital_vol()

        for iteration_count in range(1, 201):
            BSModel = BlackScholes(S=self.S, K=self.K, r=self.r, q=self.q, t=self.t, vol=implied_vol,
                                   isCall=self.isCall)
            # Calculate price of option
            calculated_price = BSModel.calculate_option_price()
            # Calculate Vega
            vega = BSModel.calculate_vega() * 100

            # Difference between the market price and calculated price
            diff = abs(calculated_price - self.price)

            # Update the implied vol as a result
            implied_vol = implied_vol - diff / vega

            # Exit for loop if difference below tolerance
            if diff <= tolerance:
                break

        # handle negative or infinite IVs
        if implied_vol < 0 or np.isnan(implied_vol):
            implied_vol = 0

        return implied_vol
