# YOUR CODE HERE
# Read the number of weeks
N = int(input())

# Read all the steps
steps = list(map(int, input().split()))

# Calculate the sum for each week
weekly_sums = []
for i in range(N):
    # For week i, we need to sum days from 7*i to 7*i+6 (inclusive)
    week_sum = sum(steps[7*i:7*i+7])
    weekly_sums.append(week_sum)

# Print the results
print(" ".join(map(str, weekly_sums)))