def can_schedule_holidays(N, A, B, D):
    # Total days in a week
    total_days = A + B
    
    # Check each plan
    for d in D:
        # Calculate the day of the week for the plan
        day_of_week = d % total_days
        # If day_of_week is 0, it means it's the last day of the week
        if day_of_week == 0:
            day_of_week = total_days
        
        # Check if the day falls within the holiday range
        if day_of_week > A:
            return "No"
    
    return "Yes"

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = int(data[1])
B = int(data[2])
D = list(map(int, data[3:]))

# Get the result
result = can_schedule_holidays(N, A, B, D)

# Print the result
print(result)