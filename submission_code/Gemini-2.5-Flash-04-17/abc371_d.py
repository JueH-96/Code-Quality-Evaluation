# YOUR CODE HERE
import sys
import bisect

# Read N
N = int(sys.stdin.readline())

# Read X coordinates
# X is guaranteed to be sorted
X = list(map(int, sys.stdin.readline().split()))

# Read P populations
P = list(map(int, sys.stdin.readline().split()))

# Calculate prefix sums of P
# S[i] will store the sum of P[0]...P[i-1]
# S has size N+1, where S[0] = 0.
# S[i+1] = S[i] + P[i] for i = 0 to N-1.
S = [0] * (N + 1)
for i in range(N):
    S[i+1] = S[i] + P[i]

# Read Q
Q = int(sys.stdin.readline())

# Process queries
for _ in range(Q):
    L, R = map(int, sys.stdin.readline().split())

    # Find the index of the first village with coordinate >= L
    # bisect_left finds the insertion point for L to maintain order,
    # which is the index of the first element >= L.
    # This index is the 0-based start index for the population sum range.
    start_idx = bisect.bisect_left(X, L)

    # Find the index of the first village with coordinate > R
    # bisect_right finds the insertion point for R to maintain order,
    # which is the index of the first element > R.
    # This index is one past the 0-based end index for the population sum range.
    idx_after_end = bisect.bisect_right(X, R)

    # The villages whose coordinates X_j are in the range [L, R]
    # correspond to indices j such that start_idx <= j < idx_after_end.
    # The sum of populations P[j] for these indices is the sum from P[start_idx]
    # up to P[idx_after_end - 1].
    # Using the prefix sum array S (where S[i] = sum(P[0]...P[i-1])),
    # this sum is S[idx_after_end] - S[start_idx].
    # This works correctly even if start_idx >= idx_after_end (empty range),
    # resulting in S[k] - S[k] = 0.
    total_population = S[idx_after_end] - S[start_idx]

    print(total_population)