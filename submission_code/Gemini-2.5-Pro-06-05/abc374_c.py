import sys

# Read the number of departments from standard input
try:
    N = int(sys.stdin.readline())
except (ValueError, IndexError):
    # Exit if input is empty or invalid
    sys.exit()

# Read the number of people in each department
K = list(map(int, sys.stdin.readline().split()))

# Calculate the total number of people across all departments
total_people = sum(K)

# This is a partition problem. We want to find a subset of departments
# whose sum of people is `s`, such that `max(s, total_people - s)` is minimized.
# This is equivalent to finding a subset sum `s` that is as close as
# possible to `total_people / 2`.

# We can find all possible subset sums using a dynamic approach.
# `possible_sums` will store all achievable sums. We start with 0 (from the empty set).
possible_sums = {0}

# Iterate through each department's size
for k_i in K:
    # For each sum `s` that is already possible, we can now also form `s + k_i`.
    # We add all such newly formed sums to our set of possibilities.
    # A set comprehension is a concise and efficient way to do this.
    possible_sums.update({s + k_i for s in possible_sums})

# To minimize `max(s, total_people - s)`, we need `s` to be close to `total_people / 2`.
# We find the largest possible sum `s` that is less than or equal to `total_people / 2`.
# Let this be `s_best`. The partition will be `s_best` and `total_people - s_best`.
# The maximum of these two is `total_people - s_best`, because `s_best <= total_people / 2`.
# This value is the minimum possible maximum.

# The ideal sum for a group is half of the total.
target = total_people // 2

# Find the largest possible sum that does not exceed the target.
s_best = 0
for s in possible_sums:
    if s <= target:
        s_best = max(s_best, s)

# The minimum possible value for the maximum group size is the larger part of the partition
# formed by `s_best`.
print(total_people - s_best)