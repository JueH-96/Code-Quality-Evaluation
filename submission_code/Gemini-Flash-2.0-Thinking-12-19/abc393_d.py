import sys

# Set recursion depth if necessary, although not needed for this problem
# sys.setrecursionlimit(2000)

def solve():
    # Read input
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # Find 0-indexed positions of all '1's
    p_indices = [i for i, char in enumerate(S) if char == '1']
    k = len(p_indices)

    # If all characters are '1's or '0's, they are already contiguous.
    # Problem guarantees at least one '1', so k >= 1.
    # If k == N, all are '1's, already contiguous. Cost is 0.
    # The case k == 0 is impossible due to constraints.
    if k == N:
        print(0)
        return

    # Calculate q_i = p_i - i for i = 0, ..., k-1
    # This transformation helps simplify the cost function.
    # The minimum number of adjacent swaps to move items from positions p_0, ..., p_{k-1}
    # to target positions t_0, ..., t_{k-1} (preserving relative order) by swapping
    # with other items is Sum_{i=0}^{k-1} |p_i - t_i|.
    # In our case, the target positions for the k '1's forming a contiguous block starting at t
    # are t, t+1, ..., t+k-1.
    # The i-th '1' (0-indexed among '1's) moves from its original position p_i to the target position t+i.
    # The cost is Sum_{i=0}^{k-1} |p_i - (t+i)|.
    # This can be rewritten as Sum_{i=0}^{k-1} |(p_i - i) - t|.
    # Let q_i = p_i - i. The cost for a given target block starting at t is Sum_{i=0}^{k-1} |q_i - t|.
    # This sum is minimized when t is the median of the q_i values.

    q_values = [p_indices[i] - i for i in range(k)]

    # The list q_values is already sorted (non-decreasing) because p_indices is sorted
    # and p_{i+1} - p_i >= 1, which implies (p_{i+1} - (i+1)) - (p_i - i) = p_{i+1} - p_i - 1 >= 0.

    # The median is the element at index k // 2 (using integer division for 0-indexed list).
    # This is the optimal value for t that minimizes Sum |q_i - t|.
    # The possible range for the starting position t is [0, N-k].
    # The values of q_i are in the range [0, N-k]:
    # Minimum q_0 = p_0 - 0 >= 0.
    # Maximum q_{k-1} = p_{k-1} - (k-1). Since p_{k-1} <= N-1, q_{k-1} <= N-1 - (k-1) = N-k.
    # So, 0 <= q_i <= N-k for all i.
    # The median of values in [0, N-k] is also within [0, N-k].
    # Thus, the optimal starting position 't' is simply the median of q_values.
    optimal_start = q_values[k // 2]

    # Calculate prefix sums of q_values for efficient computation of the sum of absolute differences.
    # P_q[j] = sum(q_values[i] for i in range(j + 1))
    prefix_sum_q = [0] * k
    if k > 0: # k >= 1 based on problem guarantees
        prefix_sum_q[0] = q_values[0]
        for i in range(1, k):
            prefix_sum_q[i] = prefix_sum_q[i-1] + q_values[i]

    # The cost is Sum_{i=0}^{k-1} |q_i - optimal_start|
    # Since q_values is sorted and optimal_start = q_values[k//2]:
    # Cost = Sum_{i=0}^{k//2} (optimal_start - q_i) + Sum_{i=k//2+1}^{k-1} (q_i - optimal_start)
    mid = k // 2

    # Sum_{i=0}^{mid} (optimal_start - q_i) = (mid + 1) * optimal_start - Sum_{i=0}^{mid} q_i
    # Sum_{i=0}^{mid} q_i is prefix_sum_q[mid]
    sum_left_part = (mid + 1) * optimal_start - prefix_sum_q[mid]

    # Sum_{i=mid+1}^{k-1} (q_i - optimal_start) = Sum_{i=mid+1}^{k-1} q_i - (number of terms) * optimal_start
    # Number of terms = (k-1) - (mid+1) + 1 = k - mid - 1
    # Sum_{i=mid+1}^{k-1} q_i = prefix_sum_q[k-1] - prefix_sum_q[mid]
    num_right_terms = k - (mid + 1)
    
    # The sum of the right part (from index mid+1 to k-1) is 0 if there are no terms (i.e., num_right_terms <= 0).
    # This happens naturally when k=1 (mid=0, num_right_terms = 1-0-1=0).
    sum_right_part = (prefix_sum_q[k-1] - prefix_sum_q[mid]) - num_right_terms * optimal_start

    cost = sum_left_part + sum_right_part

    # Python automatically uses arbitrary precision integers for large numbers.
    print(cost)

# Execute the solve function
solve()