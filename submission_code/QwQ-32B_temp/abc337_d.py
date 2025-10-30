import sys

def main():
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    min_ops = float('inf')
    
    # Process rows for horizontal possibilities
    for row in grid:
        W_row = len(row)
        x_prefix = [0] * W_row
        dot_prefix = [0] * W_row
        x_prefix[0] = 1 if row[0] == 'x' else 0
        dot_prefix[0] = 1 if row[0] == '.' else 0
        for i in range(1, W_row):
            x_prefix[i] = x_prefix[i-1] + (1 if row[i] == 'x' else 0)
            dot_prefix[i] = dot_prefix[i-1] + (1 if row[i] == '.' else 0)
        
        max_j = W_row - K
        if max_j >= 0:
            for j in range(0, max_j + 1):
                end = j + K - 1
                if j == 0:
                    x_count = x_prefix[end]
                else:
                    x_count = x_prefix[end] - x_prefix[j-1]
                if x_count == 0:
                    if j == 0:
                        dots = dot_prefix[end]
                    else:
                        dots = dot_prefix[end] - dot_prefix[j-1]
                    if dots < min_ops:
                        min_ops = dots
    
    # Process columns for vertical possibilities
    for c in range(W):
        x_prefix = [0] * H
        dot_prefix = [0] * H
        first_char = grid[0][c]
        x_prefix[0] = 1 if first_char == 'x' else 0
        dot_prefix[0] = 1 if first_char == '.' else 0
        for i in range(1, H):
            current_char = grid[i][c]
            x_prefix[i] = x_prefix[i-1] + (1 if current_char == 'x' else 0)
            dot_prefix[i] = dot_prefix[i-1] + (1 if current_char == '.' else 0)
        
        max_i = H - K
        if max_i >= 0:
            for i in range(0, max_i + 1):
                end = i + K - 1
                if i == 0:
                    x_count = x_prefix[end]
                else:
                    x_count = x_prefix[end] - x_prefix[i-1]
                if x_count == 0:
                    if i == 0:
                        dots = dot_prefix[end]
                    else:
                        dots = dot_prefix[end] - dot_prefix[i-1]
                    if dots < min_ops:
                        min_ops = dots
    
    print(-1 if min_ops == float('inf') else min_ops)

if __name__ == "__main__":
    main()