"""
Functions for formatting time
"""

import pandas as pd


def timedelta_to_years(start_date: pd.DateTime, end_date: pd.DateTime):
    """
    Get time between two datetime objects in years

    :param start_date: Start date
    :param end_date: End date
    :return: Difference between two dates in years
    """
    time_to_expiry_timedelta = end_date - start_date
    return time_to_expiry_timedelta.total_seconds() / (60 * 60 * 24 * 365)
