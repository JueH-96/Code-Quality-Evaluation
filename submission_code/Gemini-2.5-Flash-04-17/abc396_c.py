import sys

# Read input using sys.stdin.readline for speed
# N: number of black balls, M: number of white balls
N, M = map(int, sys.stdin.readline().split())

# Read values of black balls and white balls
B = list(map(int, sys.stdin.readline().split()))
W = list(map(int, sys.stdin.readline().split()))

# Sort balls in descending order to easily pick the largest values
# We want to maximize the sum, so picking larger values is generally better.
B.sort(reverse=True)
W.sort(reverse=True)

# Compute prefix sums for sorted black balls
# ps_b[k] will store the sum of the top k black ball values (indices 0 to k-1)
# ps_b[0] = 0 (sum of 0 balls)
ps_b = [0] * (N + 1)
for i in range(N):
    ps_b[i + 1] = ps_b[i] + B[i]

# Compute prefix sums for sorted white balls
# ps_w[k] will store the sum of the top k white ball values (indices 0 to k-1)
# ps_w[0] = 0 (sum of 0 balls)
ps_w = [0] * (M + 1)
for i in range(M):
    ps_w[i + 1] = ps_w[i] + W[i]

# Compute suffix maximums for prefix sums of black balls
# max_ps_b[j] will store max_{j <= k <= N} ps_b[k]
# This represents the maximum possible sum of black balls when we are constrained
# to choose at least j black balls (up to N black balls).
max_ps_b = [0] * (N + 1)

# The maximum sum choosing at least N black balls is just the sum of all N black balls
if N > 0: # Handle case N=0
    max_ps_b[N] = ps_b[N]

# Compute suffix maximums from N-1 down to 0
# max_ps_b[j] is the maximum of ps_b[j] (choosing exactly j black balls)
# and max_ps_b[j+1] (maximum sum choosing at least j+1 black balls).
# The loop range is correct even if N=0 (range(0-1, -1, -1) is empty).
for j in range(N - 1, -1, -1):
    max_ps_b[j] = max(ps_b[j], max_ps_b[j + 1])

# Initialize maximum sum found so far.
# Choosing no balls (0 black, 0 white) is always a valid option (0 >= 0)
# and gives a sum of ps_b[0] + ps_w[0] = 0 + 0 = 0.
# This covers the case where all possible positive number of balls lead to
# a negative or zero sum, in which case choosing nothing is optimal.
max_sum = 0

# Iterate through all possible numbers of white balls chosen (n_w)
# n_w can range from 0 to M
for n_w in range(M + 1):
    # If we choose n_w white balls, the sum from white balls is ps_w[n_w].
    # The problem constraint requires choosing n_b black balls such that n_b >= n_w.
    # Also, the number of black balls cannot exceed the total available N, so n_b <= N.
    # Thus, for a fixed n_w, the number of black balls n_b must be in the range [n_w, N].

    # We need to check if it's possible to choose at least n_w black balls from N balls.
    # This is only possible if n_w <= N.
    if n_w <= N:
        # If n_w <= N, the number of black balls n_b must be in the range [n_w, N].
        # To maximize the total sum for this fixed n_w, we need to choose n_b black balls
        # such that their sum ps_b[n_b] is maximized, subject to n_b in [n_w, N].
        # The maximum possible sum of black balls in this range [n_w, N] is given by max_ps_b[n_w].

        # The total sum for choosing n_w white balls and the best possible number
        # of black balls (at least n_w) is ps_w[n_w] + max_ps_b[n_w].
        current_sum = ps_w[n_w] + max_ps_b[n_w]

        # Update the overall maximum sum found so far
        max_sum = max(max_sum, current_sum)

    # If n_w > N, it is impossible to choose n_b black balls such that n_b >= n_w,
    # because the maximum number of black balls is N.
    # Thus, any choice involving more than N white balls cannot satisfy the constraint,
    # and we simply skip these iterations.

# After iterating through all valid possibilities for n_w and finding the best
# corresponding number of black balls, max_sum holds the overall maximum sum.
# Print the final maximum sum
print(max_sum)