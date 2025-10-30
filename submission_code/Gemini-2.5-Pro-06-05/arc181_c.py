import sys

def solve():
    """
    Reads input, solves the grid filling problem, and prints the resulting grid.
    """
    try:
        # Read problem inputs
        N = int(sys.stdin.readline())
        P = [int(x) - 1 for x in sys.stdin.readline().split()]
        Q = [int(x) - 1 for x in sys.stdin.readline().split()]
    except (IOError, ValueError):
        # Gracefully handle empty or malformed input at the end of a file
        return

    # Compute 0-based rank arrays.
    # rank_P[i] stores the rank of row i.
    # If P = (P_0, P_1, ...), row P_k has rank k. So, rank_P[P_k] = k.
    rank_P = [0] * N
    for k, p_val in enumerate(P):
        rank_P[p_val] = k

    # Similarly for column ranks.
    rank_Q = [0] * N
    for l, q_val in enumerate(Q):
        rank_Q[q_val] = l

    # Construct the grid based on the sum of ranks.
    # A[i][j] = '1' if rank_P[i] + rank_Q[j] >= N - 1, else '0'.
    # This construction is guaranteed to produce a valid grid.
    # Another valid threshold is N, leading to a different valid grid.
    grid_rows = []
    for i in range(N):
        row_str_list = []
        for j in range(N):
            if rank_P[i] + rank_Q[j] >= N - 1:
                row_str_list.append('1')
            else:
                row_str_list.append('0')
        grid_rows.append("".join(row_str_list))

    # Print the final grid
    print("
".join(grid_rows))

# Running the solution
solve()