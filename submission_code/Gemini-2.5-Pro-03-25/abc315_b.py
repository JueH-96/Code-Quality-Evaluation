# YOUR CODE HERE
import sys

def solve():
    # Read the number of months
    m = int(sys.stdin.readline())
    # Read the list of days in each month
    d = list(map(int, sys.stdin.readline().split()))

    # Calculate the total number of days in the year
    # The sum() function efficiently calculates the total
    total_days = sum(d)

    # Calculate the 1-based index of the middle day
    # The problem statement guarantees that total_days is odd.
    # If the total number of days is N (odd), the middle day is the ((N+1)/2)-th day.
    # We use integer division // to ensure the result is an integer.
    middle_day_index = (total_days + 1) // 2

    # Initialize the count of cumulative days passed before the start of the current month
    cumulative_days = 0

    # Iterate through each month using its 0-based index i
    for i in range(m):
        # month_index is the 1-based index (actual month number, e.g., 1 for January)
        month_index = i + 1
        # days_in_month is the number of days in the current month d[i]
        days_in_month = d[i]

        # Check if the middle day falls within the current month.
        # The days counted up to the end of the previous month is `cumulative_days`.
        # The days counted up to the end of the current month is `cumulative_days + days_in_month`.
        # If the `middle_day_index` is less than or equal to this sum, it means the middle day
        # belongs to the current month. Since we iterate sequentially, this is the first month
        # for which this condition holds true.
        if middle_day_index <= cumulative_days + days_in_month:
            # Found the month containing the middle day.
            target_month = month_index

            # Calculate the day number within this month.
            # The number of days passed *before* this month started is `cumulative_days`.
            # The middle day is the `middle_day_index`-th day overall.
            # So, its position within the current month (1-based) is the difference:
            # `middle_day_index - cumulative_days`.
            # For example, if `cumulative_days` is 181 and `middle_day_index` is 183,
            # the day is 183 - 181 = 2, meaning the 2nd day of the current month.
            target_day = middle_day_index - cumulative_days

            # Print the result in the required format "a b" where a is the month and b is the day.
            print(f"{target_month} {target_day}")

            # Return from the function since we have found and printed the answer.
            # No need to continue the loop.
            return

        # If the middle day is not in the current month, add the number of days in this month
        # to the `cumulative_days` count and proceed to check the next month.
        cumulative_days += days_in_month

# Call the solve function to execute the main logic of the program.
solve()