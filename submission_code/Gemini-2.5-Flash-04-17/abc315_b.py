# YOUR CODE HERE
import sys

# Read M
M = int(sys.stdin.readline())

# Read D_i values
# The second line contains M space-separated integers
D = list(map(int, sys.stdin.readline().split()))

# Calculate total days
total_days = sum(D)

# Calculate the target day index (1-based)
# The middle day is the ((TotalDays + 1) / 2)-th day
target_index = (total_days + 1) // 2

# Find the month and day
# cumulative_days will store the number of days in the months before the current one
cumulative_days = 0
for i in range(M): # i is 0-indexed month number (0 to M-1)
    current_month_days = D[i]
    
    # Check if target_index falls in the current month (month i+1, 1-based)
    # The days in month i+1 are from cumulative_days + 1 to cumulative_days + current_month_days
    # We are looking for the day number k such that cumulative_days < k <= cumulative_days + current_month_days
    if cumulative_days < target_index <= cumulative_days + current_month_days:
        # The target day is in month i+1
        month = i + 1
        # The day within this month is target_index - cumulative_days
        # If target_index is cumulative_days + 1, it's the 1st day (day = 1)
        # If target_index is cumulative_days + current_month_days, it's the last day (day = current_month_days)
        day = target_index - cumulative_days
        
        # Print the result and exit
        print(month, day)
        break # Found the day, no need to check further
    
    # If not in the current month, add its days to cumulative_days
    # For the next iteration, cumulative_days will be the total days before month i+2
    cumulative_days += current_month_days