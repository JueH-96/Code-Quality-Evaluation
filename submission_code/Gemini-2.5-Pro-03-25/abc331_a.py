# # YOUR CODE HERE
import sys

def solve():
    # Read the number of months M and days D per month from the first line of input
    # M: number of months in a year
    # D: number of days in a month
    M, D = map(int, sys.stdin.readline().split())

    # Read the current date: year y, month m, day d from the second line of input
    y, m, d = map(int, sys.stdin.readline().split())

    # Calculate the date of the next day based on the calendar rules of AtCoder Kingdom
    
    # Initialize the next date variables. We will update them based on the conditions.
    next_y = y
    next_m = m
    next_d = d

    # Check if the current day 'd' is less than the total number of days 'D' in a month
    if d < D:
        # Case 1: The current day is not the last day of the month.
        # The next day is simply the current day + 1, in the same month and year.
        next_d = d + 1
        # Month and year remain the same
        # next_m = m (already initialized)
        # next_y = y (already initialized)

    # If the current day 'd' is the last day of the month (d == D)
    else: 
        # Reset the day to 1 for the next day
        next_d = 1
        
        # Check if the current month 'm' is less than the total number of months 'M' in a year
        if m < M:
            # Case 2: The current day is the last day of the month (d == D),
            # but the current month is not the last month of the year.
            # The next day is the first day (1) of the next month (m + 1), in the same year.
            next_m = m + 1
            # Year remains the same
            # next_y = y (already initialized)
        
        # If the current month 'm' is the last month of the year (m == M)
        else:
            # Case 3: The current day is the last day of the month (d == D),
            # and the current month is the last month of the year (m == M).
            # The next day is the first day (1) of the first month (1) of the next year (y + 1).
            next_m = 1
            next_y = y + 1

    # Print the calculated next date (year, month, day) separated by spaces
    print(next_y, next_m, next_d)

# Call the solve function to execute the logic
solve()