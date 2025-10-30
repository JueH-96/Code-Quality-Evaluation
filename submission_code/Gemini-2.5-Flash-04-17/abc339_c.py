import sys

# Read N
N = int(sys.stdin.readline())

# Read A values
# A_1 A_2 ... A_N
A = list(map(int, sys.stdin.readline().split()))

# Initialize variables
# current_sum tracks the cumulative sum of changes S_i = A_1 + ... + A_i
# We start before the first stop, where the cumulative change is 0.
current_sum = 0
# min_sum tracks the minimum value of the cumulative sum seen so far, including S_0 = 0
# The initial state (before any stops) has 0 cumulative change (S_0).
min_sum = 0

# Iterate through the changes at each stop A_i
for a in A:
    # Update the cumulative sum S_i = S_{i-1} + A_i
    current_sum += a
    # Update the minimum cumulative sum seen so far: min(S_0, S_1, ..., S_i)
    min_sum = min(min_sum, current_sum)

# After the loop:
# current_sum holds the total cumulative change after all stops: S_N = A_1 + ... + A_N.
# min_sum holds the minimum value among all cumulative sums: min(S_0, S_1, ..., S_N), where S_0 = 0.

# Let P_0 be the initial number of passengers.
# The number of passengers after stop i is P_0 + S_i.
# The condition is that the number of passengers must always be non-negative:
# P_0 + S_i >= 0 for all i = 0, 1, ..., N.
# This requires P_0 >= -S_i for all i = 0, 1, ..., N.
# To satisfy this for all i, P_0 must be greater than or equal to the maximum of {-S_0, -S_1, ..., -S_N}.
# The minimum possible value for P_0 is max(-S_0, -S_1, ..., -S_N).
# This maximum value is equal to -min(S_0, S_1, ..., S_N).
# Our min_sum variable holds min(S_0, S_1, ..., S_N) after the loop.
# So, the minimum required initial number of passengers is -min_sum.
# Since S_0=0 is included in the minimum calculation, min_sum <= 0, which guarantees -min_sum >= 0.
min_initial_passengers = -min_sum

# The current number of passengers (after stop N) is P_0 + S_N.
# To find the minimum current number of passengers, we use the minimum required initial passengers.
# Min current passengers = min_initial_passengers + S_N
# S_N is the final value of current_sum.
min_current_passengers = min_initial_passengers + current_sum

# Print the result
print(min_current_passengers)