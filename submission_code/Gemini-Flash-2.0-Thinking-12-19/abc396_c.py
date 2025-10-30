import sys

# Function to solve the problem
def solve():
    # Read N and M from the first line
    line = sys.stdin.readline().split()
    N = int(line[0])
    M = int(line[1])

    # Read black ball values and sort in descending order
    B = list(map(int, sys.stdin.readline().split()))
    B.sort(reverse=True)

    # Read white ball values and sort in descending order
    W = list(map(int, sys.stdin.readline().split()))
    W.sort(reverse=True)

    # Compute prefix sums for sorted black balls
    # ps_b[i] stores the sum of the first i black balls (using 0-based indexing for B)
    # ps_b[0] = 0 (sum of 0 balls)
    ps_b = [0] * (N + 1)
    for i in range(N):
        ps_b[i+1] = ps_b[i] + B[i]

    # Compute prefix sums for sorted white balls
    # ps_w[j] stores the sum of the first j white balls (using 0-based indexing for W)
    # ps_w[0] = 0 (sum of 0 balls)
    ps_w = [0] * (M + 1)
    for j in range(M):
        ps_w[j+1] = ps_w[j] + W[j]

    # Compute suffix maximums for ps_b
    # max_ps_b[i] stores the maximum value among ps_b[i], ps_b[i+1], ..., ps_b[N]
    # This represents the maximum sum achievable by choosing at least i black balls (from the top of the sorted list)
    max_ps_b = [0] * (N + 1)
    # Base case: max_ps_b[N] = max(ps_b[b] for b in [N, N]) = ps_b[N]
    # Since constraints guarantee N >= 1, this index is valid.
    max_ps_b[N] = ps_b[N]
    # Compute backwards: max_ps_b[i] = max(ps_b[i], max_ps_b[i+1]) for i from N-1 down to 0
    # This recursive definition correctly calculates max(ps_b[b] for b in [i, N])
    for i in range(N - 1, -1, -1):
        max_ps_b[i] = max(ps_b[i], max_ps_b[i+1])

    # Calculate the maximum sum
    # We iterate through the number of white balls chosen, k.
    # Let k be the number of white balls chosen. To maximize the sum for a fixed count k,
    # we must choose the top k white balls. The sum from white balls is ps_w[k].
    # The number of white balls k must be between 0 and M (inclusive), i.e., 0 <= k <= M.
    # If we choose k white balls, we must choose b black balls such that the number of
    # black balls b is greater than or equal to the number of white balls k, i.e., b >= k.
    # The number of black balls b must also be at most N, i.e., 0 <= b <= N.
    # To maximize the sum from black balls for a fixed count b, we must choose the top b
    # black balls. The sum from black balls is ps_b[b].
    # So, for a fixed number of white balls k, we must choose b black balls such that k <= b <= N.
    # We want to maximize ps_w[k] + ps_b[b] over all possible pairs (b, k) that satisfy the constraints.
    # We can iterate over the number of white balls chosen, k, from 0 to M.
    # For each k, we must choose b such that k <= b <= N. This is only possible if k <= N.
    # So we only need to consider k in the range [0, min(M, N)].
    # For a fixed k in this range, the sum from white balls is ps_w[k].
    # We need to choose b black balls such that k <= b <= N to maximize ps_b[b].
    # The maximum value of ps_b[b] for b in the range [k, N] is given by max_ps_b[k].
    # So, for a fixed k, the maximum possible sum is ps_w[k] + max_ps_b[k].
    # We take the maximum of this sum over all valid values of k.

    # Initialize max_sum to 0. This handles the case of choosing no balls (0 black, 0 white),
    # which results in a sum of 0. This is a valid choice (0 >= 0).
    max_sum = 0

    # Iterate k from 0 up to min(M, N) inclusive.
    # For each k, we consider choosing the top k white balls.
    # The number of white balls chosen must be k.
    for k in range(min(M, N) + 1):
        # current_sum is the maximum sum when choosing k white balls (top k)
        # and at least k black balls (top b, where b >= k). The maximum sum from
        # choosing at least k black balls is max_ps_b[k].
        current_sum = ps_w[k] + max_ps_b[k]
        # Update the overall maximum sum found so far
        max_sum = max(max_sum, current_sum)

    # Print the result to standard output
    sys.stdout.write(str(max_sum) + '
')

# Call the solve function to run the program
solve()