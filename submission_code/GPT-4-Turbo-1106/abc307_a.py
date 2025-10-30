# Read the number of weeks from standard input
N = int(input().strip())

# Read the steps for each day, split by space, and convert to integers
steps = list(map(int, input().strip().split()))

# Initialize an empty list to store the total steps for each week
weekly_steps = []

# Calculate the total steps for each week
for i in range(N):
    # Sum the steps for the current week (7 days)
    week_sum = sum(steps[i*7:(i+1)*7])
    # Append the sum to the weekly_steps list
    weekly_steps.append(week_sum)

# Print the weekly steps separated by space
print(' '.join(map(str, weekly_steps)))