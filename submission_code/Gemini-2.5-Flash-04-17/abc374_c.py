import sys

# Read N
N = int(sys.stdin.readline())

# Read K_i values
K = list(map(int, sys.stdin.readline().split()))

# Calculate the total sum of all people
total_sum = sum(K)

# Initialize the minimum possible value of the maximum sum
# The maximum sum is at most the total sum (when one group is empty)
# Initialize with total_sum is safe because the partition (all to one group) is always possible
min_max_sum = total_sum

# Iterate through all 2^N possible ways to assign departments to groups.
# We use a bitmask approach. Each integer `mask` from 0 to 2^N - 1
# represents a unique partition of the N departments into two groups.
# If the j-th bit of `mask` is 1, we assign department j (0-indexed) to Group A.
# If the j-th bit of `mask` is 0, we assign department j to Group B.
# This approach ensures every possible partition is considered exactly once.
for mask in range(1 << N):
    sum_a = 0
    # Calculate the sum of people in Group A based on the current mask
    for j in range(N):
        # Check if the j-th bit is set in the mask
        if (mask >> j) & 1:
            # If set (bit is 1), add the population of department j to sum_a (Group A)
            sum_a += K[j]

    # The sum of people in Group B is the total sum minus the sum of Group A.
    # This implicitly assigns departments where the j-th bit was 0 to Group B.
    sum_b = total_sum - sum_a

    # Calculate the maximum sum between the two groups for this partition
    current_max_sum = max(sum_a, sum_b)

    # Update the overall minimum maximum sum found so far
    min_max_sum = min(min_max_sum, current_max_sum)

# The loop has considered all 2^N partitions.
# min_max_sum now holds the minimum possible value of the maximum number of people
# taking a lunch break at the same time.
print(min_max_sum)