import sys

# Use standard input and output
input = sys.stdin.readline

def solve():
    line1 = input().split()
    N = int(line1[0])
    M = int(line1[1])

    X = list(map(int, input().split()))
    A = list(map(int, input().split()))

    # Sort pairs based on position X_i
    initial_pairs = sorted(zip(X, A))

    # Check total stones
    total_stones = sum(A)
    if total_stones != N:
        print(-1)
        return

    # Calculate sum_{k=1}^{N-1} (C_k - k) where C_k = sum(A_i for X_i <= k)
    # This sum represents the minimum number of operations if possible.
    # We also check the necessary condition C_k >= k for 1 <= k <= N-1.

    current_cumulative_A = 0 # Represents sum(A_i for X_i <= prev_x)
    total_ops = 0
    prev_x = 0 # Represents the last processed position X_i, or 0 initially

    # Iterate through the sorted initial positions and amounts
    # The initial positions divide the range [1, N] into segments.
    # We calculate the contribution to total_ops from each segment and the points X_i.
    for i in range(M):
        x_i, a_i = initial_pairs[i]

        # Process the range of cells (k) from prev_x + 1 up to x_i - 1.
        # This is the gap between the previous initial stone position and the current one.
        # For any k in this gap (prev_x + 1 <= k <= x_i - 1), the cumulative stone count C_k
        # is constant and equal to the total stones from initial positions strictly less than x_i,
        # which is exactly `current_cumulative_A`.
        start_k = prev_x + 1
        end_k = x_i - 1

        if start_k <= end_k: # Check if the gap is non-empty (i.e., x_i > prev_x + 1)
            # The necessary condition C_k >= k must hold for all k in this gap.
            # C_k = current_cumulative_A. The condition current_cumulative_A >= k is most restrictive
            # for the largest k in the gap, which is end_k (x_i - 1).
            # If current_cumulative_A < x_i - 1, it's impossible to have C_k >= k for k = x_i - 1.
            if current_cumulative_A < end_k:
                print(-1)
                return

            # Calculate the contribution to total operations from this gap.
            # This is sum_{k=start_k}^{end_k} (C_k - k) = sum_{k=start_k}^{end_k} (current_cumulative_A - k)
            # Using arithmetic series sum formula: sum(c - k for k=L to R) = c * (R - L + 1) - sum(k for k=L to R)
            # sum(k for k=L to R) = (L + R) * (R - L + 1) / 2
            num_terms = end_k - start_k + 1
            sum_of_k_in_gap = (start_k + end_k) * num_terms // 2
            total_ops += num_terms * current_cumulative_A - sum_of_k_in_gap

        # Process the position x_i itself.
        # At k = x_i, the cumulative stone count C_{x_i} is current_cumulative_A (stones before x_i) + a_i (stones at x_i).
        # Update current_cumulative_A to include stones at x_i.
        current_cumulative_A += a_i

        # Check the necessary condition C_{x_i} >= x_i
        if current_cumulative_A < x_i:
            print(-1)
            return

        # The flow across the boundary x_i -> x_i+1 is C_{x_i} - x_i.
        # This contributes C_{x_i} - x_i to the total operations.
        # This represents the sum (C_k - k) for k = x_i.
        total_ops += (current_cumulative_A - x_i)

        # Update prev_x to the current position x_i
        prev_x = x_i

    # After iterating through all initial stone positions, handle the final range of cells
    # from x_M + 1 up to N - 1.
    # For k in this final gap (x_M + 1 <= k <= N - 1), the cumulative stone count C_k is constant,
    # and equal to the total stones from all initial positions <= x_M, which is sum(A_i) = N.
    start_k = prev_x + 1 # This will be x_M + 1
    end_k = N - 1

    if start_k <= end_k: # Check if the final gap is non-empty (i.e., x_M < N - 1)
        # The necessary condition C_k >= k must hold for all k in this gap.
        # C_k = N. The condition N >= k is true for all k in this gap since k <= N - 1. Condition holds.

        # Calculate the contribution to total operations from this final gap.
        # This is sum_{k=start_k}^{end_k} (C_k - k) = sum_{k=start_k}^{end_k} (N - k)
        num_terms = end_k - start_k + 1
        sum_of_k_in_gap = (start_k + end_k) * num_terms // 2
        # C_k is N in this range, use N explicitly.
        total_ops += num_terms * N - sum_of_k_in_gap

    # If all checks passed, total_ops contains the minimum number of operations.
    print(total_ops)

solve()