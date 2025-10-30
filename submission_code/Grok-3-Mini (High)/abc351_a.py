import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Sum the first nine elements for Takahashi's total score
takahashi_sum = sum(int(data[i]) for i in range(9))

# Sum the next eight elements for Aoki's current score (first eight innings)
aoki_sum_current = sum(int(data[i]) for i in range(9, 17))

# Calculate the difference
difference = takahashi_sum - aoki_sum_current

# Minimum runs needed for Aoki to win is difference + 1
min_runs_needed = difference + 1

# Output the result
print(min_runs_needed)