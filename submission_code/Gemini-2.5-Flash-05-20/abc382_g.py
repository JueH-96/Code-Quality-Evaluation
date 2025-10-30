import sys

def solve():
    K, S_x, S_y, T_x, T_y = map(int, sys.stdin.readline().split())

    # Function to get (i, j, k) coordinates for a given (X, Y) and K_val.
    # The X, Y inputs here are the integer coordinates before adding 0.5.
    def get_tile_coords(X, Y, K_val):
        i = X // K_val
        j = Y // K_val
        
        # Determine tile type based on parity of i and j.
        # (i % 2) and (j % 2) correctly handle negative numbers in Python.
        if (i % 2) == (j % 2): # H-tile: i and j have the same parity
            # For H-tile, k is based on the Y-coordinate's remainder within its K-block.
            k = Y % K_val
        else: # V-tile: i and j have different parity
            # For V-tile, k is based on the X-coordinate's remainder within its K-block.
            k = X % K_val
        return (i, j, k)

    # Get the (i, j, k) coordinates for the starting and target tiles.
    si, sj, sk = get_tile_coords(S_x, S_y, K)
    ti, tj, tk = get_tile_coords(T_x, T_y, K)

    ans = 0

    # Case 1: Start and target are in the same major (i,j) block.
    if si == ti and sj == tj:
        # If in the same block, only k needs to change.
        ans = abs(sk - tk)
    # Case 2: Start and target are in different major (i,j) blocks.
    else:
        # Calculate the Manhattan distance in terms of major (i,j) blocks.
        # Each step in this distance represents a transition between major blocks,
        # which costs 1 move.
        dist_i = abs(si - ti)
        dist_j = abs(sj - tj)
        D = dist_i + dist_j # Total number of major block transitions required.

        # For a major block transition to occur, the k-coordinate of the source tile
        # must be either 0 or K-1.
        # Calculate the minimum moves required to bring the starting k (sk)
        # to one of these boundary values (0 or K-1).
        # If sk is already 0 or K-1, this cost is 0.
        cost_to_boundary_sk = min(sk, K - 1 - sk)
        
        # The total cost is:
        # 1. The initial moves to bring sk to a boundary (0 or K-1) if it's not already.
        # 2. D moves for the major block transitions.
        # During these D transitions, each move costs 1. The rule that we can land
        # on any k-value (0 to K-1) in the destination tile means we can always
        # choose the k-value of the final destination tile (tk) for the last major
        # block transition without any additional cost. Similarly, for intermediate
        # transitions, we can choose to land on 0 or K-1 for the next transition.
        ans = cost_to_boundary_sk + D
    
    sys.stdout.write(str(ans) + '
')

# Read the number of test cases.
num_test_cases = int(sys.stdin.readline())

# Process each test case.
for _ in range(num_test_cases):
    solve()