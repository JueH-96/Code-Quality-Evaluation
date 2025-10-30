# Read N, the number of weeks
N = int(input())

# Read the list of all daily steps
# The input for steps is given as a single line of space-separated integers
A = list(map(int, input().split()))

# Initialize an empty list to store the sum of steps for each week
weekly_sums = []

# Iterate N times, once for each week
for i in range(N):
    # Calculate the starting index for the current week's days
    # Each week has 7 days. For the i-th week (0-indexed), the days start at index i * 7
    start_index = i * 7

    # Calculate the ending index for the current week's days (exclusive)
    # The days for the current week are from start_index to start_index + 7 - 1
    # So, the slice will be A[start_index : start_index + 7]
    end_index = start_index + 7

    # Extract the steps for the current week using slicing
    current_week_steps = A[start_index:end_index]

    # Calculate the sum of steps for the current week
    week_total = sum(current_week_steps)

    # Add the weekly total to our list of weekly sums
    weekly_sums.append(week_total)

# Print the weekly sums, separated by spaces
# The * operator unpacks the list into separate arguments for the print function
print(*weekly_sums)