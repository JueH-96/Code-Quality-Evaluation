from itertools import groupby

def solve(H, W, K, S):
    # Count consecutive 'o' and '.' in each row and column
    row_counts = [[(k, len(list(v))) for k, v in groupby(s)] for s in S]
    col_counts = [[(k, len(list(v))) for k, v in groupby(''.join([S[i][j] for i in range(H)]))] for j in range(W)]

    # Check for existing K consecutive 'o'
    for counts in row_counts + col_counts:
        for k, length in counts:
            if k == 'o' and length >= K:
                return 0

    # Calculate minimum operations for each row and column
    min_ops = float('inf')
    for counts in row_counts + col_counts:
        ops = 0
        consecutive_o = 0
        for k, length in counts:
            if k == 'o':
                consecutive_o += length
                if consecutive_o >= K:
                    min_ops = min(min_ops, ops)
            else:
                if consecutive_o >= K:
                    min_ops = min(min_ops, ops)
                consecutive_o = 0
                ops += length

    # If no solution found, return -1
    return min_ops if min_ops != float('inf') else -1

# Read input
H, W, K = map(int, input().split())
S = [input() for _ in range(H)]

# Solve and print the answer
print(solve(H, W, K, S))