# YOUR CODE HERE
import sys

# Read N
N = int(sys.stdin.readline().strip())

# Read D_i values
# D is a list where D[i-1] is the number of days in month i
D = list(map(int, sys.stdin.readline().strip().split()))

# Initialize count
repdigit_count = 0

# Iterate through possible repeating digits d (1 to 9)
# A repdigit number cannot use the digit 0 if it's a month or day number (which are >= 1)
for d in range(1, 10):
    # Potential month numbers using digit d: d and dd
    # Month numbers are 1-indexed.
    month_d = d
    month_dd = d * 10 + d # e.g., 11, 22, ..., 99

    # Potential day numbers using digit d: d and dd
    # Day numbers are 1-indexed.
    day_d = d
    day_dd = d * 10 + d # e.g., 11, 22, ..., 99

    # Check if month 'd' exists (1 <= d <= N)
    if month_d <= N:
        # Month 'd' exists. Get the number of days in month 'd'.
        # Access D using 0-indexed month number (month - 1).
        days_in_this_month = D[month_d - 1]

        # Check if day 'd' is valid in month 'd' (1 <= d <= days_in_this_month)
        if day_d <= days_in_this_month:
            repdigit_count += 1 # Date (month_d, day_d) is a repdigit date

        # Check if day 'dd' is valid in month 'd' (1 <= dd <= days_in_this_month)
        # day_dd for d=1..9 ranges from 11 to 99. Constraints state D_i <= 100,
        # so we only need to check if day_dd <= days_in_this_month.
        # The smallest day_dd is 11 (for d=1). The smallest day_d is 1 (for d=1).
        # Both are >= 1.
        if day_dd <= days_in_this_month:
             repdigit_count += 1 # Date (month_d, day_dd) is a repdigit date

    # Check if month 'dd' exists (1 <= dd <= N)
    # month_dd for d=1..9 ranges from 11 to 99. Constraints state N <= 100,
    # so month_dd can exist.
    if month_dd <= N:
        # Month 'dd' exists. Get the number of days in month 'dd'.
        # Access D using 0-indexed month number (month - 1).
        days_in_this_month = D[month_dd - 1]

        # Check if day 'd' is valid in month 'dd' (1 <= d <= days_in_this_month)
        if day_d <= days_in_this_month:
            repdigit_count += 1 # Date (month_dd, day_d) is a repdigit date

        # Check if day 'dd' is valid in month 'dd' (1 <= dd <= days_in_this_month)
        # day_dd for d=1..9 ranges from 11 to 99. Constraints state D_i <= 100,
        # so we only need to check if day_dd <= days_in_this_month.
        if day_dd <= days_in_this_month:
            repdigit_count += 1 # Date (month_dd, day_dd) is a repdigit date

# Print the total count
print(repdigit_count)