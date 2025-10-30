import sys

def solve():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))

    # Create rank_P_arr: rank_P_arr[row_idx] stores the 1-indexed rank of row_idx in permutation P.
    # For example, if P = (1, 3, 2) and N=3:
    # P_1=1, P_2=3, P_3=2 (1-indexed values)
    # The actual row indices (0-indexed) are 0, 1, 2.
    # Row P[0]-1 (which is 0) has rank 1 (since it's P_1).
    # Row P[1]-1 (which is 2) has rank 2 (since it's P_2).
    # Row P[2]-1 (which is 1) has rank 3 (since it's P_3).
    # So, rank_P_arr[0] = 1, rank_P_arr[2] = 2, rank_P_arr[1] = 3.
    rank_P_arr = [0] * N
    for k in range(N):
        # P[k] is the (k+1)-th element in the P sequence (1-indexed value P_k from problem).
        # Convert it to 0-indexed row: P[k]-1.
        # Its rank is k+1 (since it's the (k+1)-th element in the ordering).
        rank_P_arr[P[k]-1] = k + 1

    # Create rank_Q_arr similarly for columns.
    rank_Q_arr = [0] * N
    for k in range(N):
        rank_Q_arr[Q[k]-1] = k + 1

    # Initialize the N-by-N grid with zeros.
    grid = [[0] * N for _ in range(N)]

    # Fill the grid based on the derived condition: A[i][j] = 1 if rank_P_arr[i] + rank_Q_arr[j] > N, else 0.
    for i in range(N):
        for j in range(N):
            # i is the 0-indexed row number
            # j is the 0-indexed column number
            if rank_P_arr[i] + rank_Q_arr[j] > N:
                grid[i][j] = 1
            else:
                grid[i][j] = 0

    # Print the grid.
    # Each row should be printed as a string of '0's and '1's.
    for i in range(N):
        print("".join(map(str, grid[i])))

solve()