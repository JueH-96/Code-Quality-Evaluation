import sys

def main():
    N = int(sys.stdin.readline())
    P_in = list(map(int, sys.stdin.readline().split()))
    Q_in = list(map(int, sys.stdin.readline().split()))

    # rank_P[i] stores the 0-indexed rank of row i.
    # A row's rank is its position (0 to N-1) in the desired lexicographical order.
    # P_in is 1-indexed. P_in[k] is the 1-based index of the row that has rank k (0-indexed).
    # So, row P_in[k]-1 (0-indexed) has rank k.
    # rank_P[ (P_in[k]-1) ] = k
    rank_P = [0] * N
    for k in range(N): # k is the 0-indexed rank
        row_idx_1_based = P_in[k]
        rank_P[row_idx_1_based - 1] = k
    
    # Similarly for columns
    rank_Q = [0] * N
    for k in range(N): # k is the 0-indexed rank
        col_idx_1_based = Q_in[k]
        rank_Q[col_idx_1_based - 1] = k

    # grid[i][j] = 1 if rank_P[i] + rank_Q[j] >= N.
    # Here, rank_P[i] and rank_Q[j] are 0-indexed ranks (0 to N-1).
    # This corresponds to rank_P_1idx[i] + rank_Q_1idx[j] > N+1 for 1-indexed ranks.
    
    output_rows = []
    for i in range(N): # 0-indexed row index
        current_row_chars = []
        for j in range(N): # 0-indexed column index
            if rank_P[i] + rank_Q[j] >= N:
                current_row_chars.append('1')
            else:
                current_row_chars.append('0')
        output_rows.append("".join(current_row_chars))
        
    sys.stdout.write("
".join(output_rows) + "
")

if __name__ == '__main__':
    main()