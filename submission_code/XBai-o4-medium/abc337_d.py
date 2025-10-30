import sys

def main():
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    min_ops = float('inf')

    # Process horizontal rows
    for row in grid:
        W_current = len(row)
        if W_current < K:
            continue
        # Compute prefix_x and prefix_dot
        prefix_x = [0] * (W_current + 1)
        prefix_dot = [0] * (W_current + 1)
        for i in range(W_current):
            prefix_x[i+1] = prefix_x[i] + (1 if row[i] == 'x' else 0)
            prefix_dot[i+1] = prefix_dot[i] + (1 if row[i] == '.' else 0)
        # Check all possible horizontal windows
        max_j = W_current - K
        for j in range(max_j + 1):
            end = j + K
            x_count = prefix_x[end] - prefix_x[j]
            if x_count == 0:
                dot_count = prefix_dot[end] - prefix_dot[j]
                if dot_count < min_ops:
                    min_ops = dot_count

    # Process vertical columns
    for j in range(W):
        column = []
        for i in range(H):
            column.append(grid[i][j])
        H_current = len(column)
        if H_current < K:
            continue
        # Compute prefix_x and prefix_dot for the column
        prefix_x = [0] * (H_current + 1)
        prefix_dot = [0] * (H_current + 1)
        for i in range(H_current):
            prefix_x[i+1] = prefix_x[i] + (1 if column[i] == 'x' else 0)
            prefix_dot[i+1] = prefix_dot[i] + (1 if column[i] == '.' else 0)
        # Check all possible vertical windows
        max_i = H_current - K
        for i in range(max_i + 1):
            end = i + K
            x_count = prefix_x[end] - prefix_x[i]
            if x_count == 0:
                dot_count = prefix_dot[end] - prefix_dot[i]
                if dot_count < min_ops:
                    min_ops = dot_count

    if min_ops == float('inf'):
        print(-1)
    else:
        print(min_ops)

if __name__ == "__main__":
    main()