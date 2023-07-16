import math
import pandas as pd
from scipy.stats import norm

N = norm.cdf
N_PRIME = norm.pdf


def datetimes_to_years(start_date: pd.DateTime, end_date: pd.DateTime):
    """
    Get time between two datetime objects in years

    :param start_date: Start date
    :param end_date: End date
    :return: Difference between two dates in years
    """
    time_to_expiry_timedelta = end_date - start_date
    return time_to_expiry_timedelta.total_seconds() / (60 * 60 * 24 * 365)


class BlackScholes:
    def __int__(self, S, K, r, q, t, vol, isCall) -> None:
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
                + (self.t * (self.r - self.q + ((self.vol ** 2)/2)))) / (self.vol * math.sqrt(self.t))

    def calculate_d2(self):
        return self.calculate_d1() - (self.vol * math.sqrt(self.t))

    def calculate_option_price(self):
        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        sign = 1 if self.isCall else -1

        return (sign * self.S * math.exp(-self.q * self.t) * N(sign * d1))\
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
        self.est_vol = est_vol
        self.price = price
        self.isCall = isCall

    def newton_raphson(self):
        tolerance = 1e-8
        x0 = self.est_vol
        x1 = 0
        BSModel = BlackScholes(S=self.S, K=self.K, r=self.r, q=self.q, t=self.t, vol=x0, isCall=self.isCall)

        fx0 = BSModel.calculate_option_price() - self.price

        vega = BSModel.calculate_vega() * 100

        while abs(x1 - x0) > tolerance:
            x1 = x0 - fx0 / vega
            BSModel = BlackScholes(S=self.S, K=self.K, r=self.r, q=self.q, t=self.t, vol=x1, isCall=self.isCall)
            fx1 = BSModel.calculate_option_price() - self.price
            vega = BSModel.calculate_vega() * 100
            x0 = x1
            fx0 = fx1

        if x1 < 0:
            x1 = 0

        return x1
