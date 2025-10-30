import sys

def main():
    H, W, K = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Precompute row prefix sums for dots and x's
    pre_row_dots = [[0]*(W+1) for _ in range(H)]
    pre_row_x = [[0]*(W+1) for _ in range(H)]
    for i in range(H):
        for j in range(W):
            pre_row_dots[i][j+1] = pre_row_dots[i][j] + (1 if S[i][j] == '.' else 0)
            pre_row_x[i][j+1] = pre_row_x[i][j] + (1 if S[i][j] == 'x' else 0)
    
    # Precompute column prefix sums for dots and x's
    pre_col_dots = [[0]*(H+1) for _ in range(W)]
    pre_col_x = [[0]*(H+1) for _ in range(W)]
    for j in range(W):
        for i in range(H):
            pre_col_dots[j][i+1] = pre_col_dots[j][i] + (1 if S[i][j] == '.' else 0)
            pre_col_x[j][i+1] = pre_col_x[j][i] + (1 if S[i][j] == 'x' else 0)
    
    min_ops = float('inf')
    
    # Check all horizontal windows
    for i in range(H):
        max_s = W - K
        if max_s >= 0:
            for s in range(max_s + 1):
                end = s + K
                x_count = pre_row_x[i][end] - pre_row_x[i][s]
                if x_count == 0:
                    dot_count = pre_row_dots[i][end] - pre_row_dots[i][s]
                    if dot_count < min_ops:
                        min_ops = dot_count
    
    # Check all vertical windows
    for j in range(W):
        max_s = H - K
        if max_s >= 0:
            for s in range(max_s + 1):
                end = s + K
                x_count = pre_col_x[j][end] - pre_col_x[j][s]
                if x_count == 0:
                    dot_count = pre_col_dots[j][end] - pre_col_dots[j][s]
                    if dot_count < min_ops:
                        min_ops = dot_count
    
    print(-1 if min_ops == float('inf') else min_ops)

if __name__ == '__main__':
    main()