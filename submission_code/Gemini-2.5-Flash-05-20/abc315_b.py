import sys

def solve():
    # Read the number of months, M
    M = int(sys.stdin.readline())

    # Read the list of days for each month, D_i
    # D will be a list of integers
    D = list(map(int, sys.stdin.readline().split()))

    # Calculate the total number of days in the year
    total_days_in_year = sum(D)

    # Calculate the index of the middle day.
    # The problem specifies ((D_1+...+D_M+1)/2)-th day.
    # Since total_days_in_year is guaranteed to be odd, total_days_in_year + 1 will be even,
    # so integer division (//) is correct here.
    target_day_index = (total_days_in_year + 1) // 2

    # Initialize a variable to keep track of days passed before the current month
    current_days_passed = 0

    # Iterate through each month to find where the target day falls
    # month_index goes from 0 to M-1, representing D[0] to D[M-1]
    for month_index in range(M):
        # Get the number of days in the current month
        days_in_current_month = D[month_index]
        
        # Check if the target day falls within this month
        # This condition means: the target_day_index is greater than the days passed
        # before this month (current_days_passed) AND less than or equal to the
        # total days passed including this month (current_days_passed + days_in_current_month).
        if current_days_passed + days_in_current_month >= target_day_index:
            # The target day is in this month.
            # Calculate the 1-indexed month number (month_a)
            month_a = month_index + 1
            
            # Calculate the 1-indexed day number within this month (day_b)
            # If current_days_passed is 181, and target_day_index is 183,
            # then day 1 of this month is the 182nd day of the year,
            # day 2 is the 183rd day of the year. So, 183 - 181 = 2.
            day_b = target_day_index - current_days_passed
            
            # Print the result in the specified format
            print(f"{month_a} {day_b}")
            
            # We found the day, so we can exit the function
            return

# Call the solve function to execute the program
solve()