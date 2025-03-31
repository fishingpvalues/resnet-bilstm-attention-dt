# TIME CONVERSION UTILITY

import math
from typing import Tuple


def sin_cos_to_weekday(sin_val: float, cos_val: float) -> Tuple[int, str]:
    """
    Given the sine and cosine values computed as:
      sin_val = sin(2*pi*weekday/7)
      cos_val = cos(2*pi*weekday/7)
    this function recovers the weekday as an integer (0=Monday, ..., 6=Sunday)
    and returns the corresponding weekday name.
    """
    # Compute the angle in radians from sin and cos values
    angle = math.atan2(sin_val, cos_val)
    # Ensure angle is in [0, 2*pi)
    if angle < 0:
        angle += 2 * math.pi
    # Invert: angle = 2*pi*day/7  =>  day = angle * 7 / (2*pi)
    day = angle * 7 / (2 * math.pi)
    # Round to nearest integer to account for floating point error,
    # and use modulo 7 in case rounding gives 7.
    weekday = int(round(day)) % 7

    weekday_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    return weekday, weekday_names.get(weekday, "Unknown")


def describe_sin_cos(sin_val: float, cos_val: float) -> str:
    """
    Returns a string describing the provided sin and cos values,
    along with the recovered weekday information.
    """
    weekday, weekday_name = sin_cos_to_weekday(sin_val, cos_val)
    return (
        f"sin: {sin_val:.4f}, cos: {cos_val:.4f} corresponds approximately "
        f"to weekday index {weekday} ({weekday_name})."
    )


from datetime import datetime

import holidays


def is_not_weekday(date: datetime) -> bool:
    """
    Check if the given date is a non-working day (weekend or German holiday).
    """
    # Check if the date is a weekend (Saturday=5, Sunday=6)
    if date.weekday() >= 5:
        return True
    # Check if the date is a holiday in Germany
    de_holidays = holidays.Germany(years=date.year)
    return date in de_holidays


def is_break(timepoint: datetime) -> bool:
    """
    Check if the given datetime falls into a break period.
    Break periods are defined as:
      - Night break: from 17:00 (inclusive) to 08:00 (exclusive)
      - Lunch break: from 12:00 (inclusive) to 14:00 (exclusive)
    """
    hour = timepoint.hour
    in_night_break = hour >= 17 or hour < 8
    in_lunch_break = 12 <= hour < 14
    return in_night_break or in_lunch_break
