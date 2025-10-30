import sys

# Read the input year
Y = int(sys.stdin.readline())

# Determine if it is a leap year based on the Gregorian calendar rules provided:
# A year is a leap year (366 days) if:
# 1. It is a multiple of 4 AND not a multiple of 100, OR
# 2. It is a multiple of 400.
# Otherwise, it is a common year (365 days).

if (Y % 4 == 0 and Y % 100 != 0) or (Y % 400 == 0):
    # It's a leap year
    days_in_year = 366
else:
    # It's a common year
    days_in_year = 365

# Print the number of days
print(days_in_year)