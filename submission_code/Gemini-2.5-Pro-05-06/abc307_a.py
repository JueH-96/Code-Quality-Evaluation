# Read the number of weeks
N = int(input())

# Read the steps for each day over 7*N days
# The input is a single line of space-separated integers
A = list(map(int, input().split()))

# Initialize a list to store the sum of steps for each week
weekly_sums = []

# Iterate through each week
# The loop variable 'i' will go from 0 to N-1, representing the i-th week (0-indexed)
for i in range(N):
    # Calculate the starting index in the list A for the current week
    # For the first week (i=0), start_index = 0
    # For the second week (i=1), start_index = 7, and so on.
    start_index = i * 7
    
    # Calculate the ending index (exclusive) in the list A for the current week
    # Each week has 7 days.
    end_index = start_index + 7
    
    # Slice the list A to get the steps for the current week
    current_week_steps = A[start_index:end_index]
    
    # Calculate the sum of steps for the current week
    current_week_total_steps = sum(current_week_steps)
    
    # Add this sum to our list of weekly sums
    weekly_sums.append(current_week_total_steps)

# Convert the list of integer sums to a list of strings
# Then, join these strings with a space in between to match the output format
output_string = " ".join(map(str, weekly_sums))

# Print the final result
print(output_string)