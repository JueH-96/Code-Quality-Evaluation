import sys

# Main function to solve the problem
def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Read initial ball counts A_i
    A = list(map(int, sys.stdin.readline().split()))

    # Read operation box indices B_i
    B_ops = list(map(int, sys.stdin.readline().split()))

    # Segment Tree implementation using a lazy propagation array
    # lazy[node_idx] stores the value to be added to all elements in the node's range.
    # The actual values are derived from initial A[pos] + sum of lazy tags on path to pos.
    # We use 4*N as a safe upper bound for the segment tree array size.
    lazy = [0] * (4 * N)

    # Helper function to push lazy tag down to children
    def push(node_idx):
        if lazy[node_idx] != 0:
            # If not a leaf node, propagate lazy tag to children
            # Children's lazy tags add the parent's lazy value
            lazy[2 * node_idx] += lazy[node_idx]
            lazy[2 * node_idx + 1] += lazy[node_idx]
            # Clear parent's lazy tag as it has been propagated
            lazy[node_idx] = 0

    # Function to add 'val' to elements in range [query_l, query_r]
    # Call with: range_add(1, 0, N-1, query_l, query_r, val)
    def range_add(node_idx, current_l, current_r, query_l, query_r, val):
        # Current segment is completely outside the query range
        if current_l > query_r or current_r < query_l:
            return

        # Current segment is completely inside the query range
        if query_l <= current_l and current_r <= query_r:
            lazy[node_idx] += val
            return

        # Current segment partially overlaps, push lazy tag down and recurse
        push(node_idx) # Push lazy tag down before recursing
        mid = (current_l + current_r) // 2
        range_add(2 * node_idx, current_l, mid, query_l, query_r, val)
        range_add(2 * node_idx + 1, mid + 1, current_r, query_l, query_r, val)

    # Function to query the value at a specific position 'pos'
    # Call with: query_point(1, 0, N-1, pos)
    def query_point(node_idx, current_l, current_r, pos):
        # If it's a leaf node, return initial value + accumulated lazy value
        if current_l == current_r:
            return A[pos] + lazy[node_idx]

        # Push lazy tag down before recursing
        push(node_idx)
        mid = (current_l + current_r) // 2
        if pos <= mid:
            return query_point(2 * node_idx, current_l, mid, pos)
        else:
            return query_point(2 * node_idx + 1, mid + 1, current_r, pos)

    # Perform M operations
    for b_idx in B_ops:
        # 1. Take out all balls from box b_idx
        balls_in_hand = query_point(1, 0, N-1, b_idx)

        # 2. Set box b_idx to 0 (by adding -balls_in_hand to it)
        range_add(1, 0, N-1, b_idx, b_idx, -balls_in_hand)

        # 3. Distribute balls if any were taken out
        if balls_in_hand > 0:
            # Distribute k // N balls to all boxes
            full_cycles = balls_in_hand // N
            if full_cycles > 0:
                range_add(1, 0, N-1, 0, N-1, full_cycles)

            # Distribute the remaining k % N balls
            rem_balls = balls_in_hand % N
            if rem_balls > 0:
                start_dist_idx = (b_idx + 1) % N
                # The last box index for the remaining balls
                end_dist_idx = (start_dist_idx + rem_balls - 1) % N

                if start_dist_idx <= end_dist_idx:
                    # No wrap-around, a single contiguous range
                    range_add(1, 0, N-1, start_dist_idx, end_dist_idx, 1)
                else:
                    # Wrap-around: distribute to (start_dist_idx to N-1) and then (0 to end_dist_idx)
                    range_add(1, 0, N-1, start_dist_idx, N-1, 1)
                    range_add(1, 0, N-1, 0, end_dist_idx, 1)

    # Collect the final number of balls in each box
    final_counts = [0] * N
    for i in range(N):
        final_counts[i] = query_point(1, 0, N-1, i)

    # Print the results separated by spaces
    sys.stdout.write(" ".join(map(str, final_counts)) + "
")

# Call the solver function
solve()