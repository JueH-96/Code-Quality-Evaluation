def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    grid = []
    for _ in range(H):
        grid.append(input[idx].strip())
        idx += 1
    
    min_ops = float('inf')
    
    # Process rows
    for row in grid:
        n = len(row)
        if K > n:
            continue
        prefix_x = [0] * (n + 1)
        prefix_dot = [0] * (n + 1)
        for i in range(n):
            prefix_x[i+1] = prefix_x[i] + (1 if row[i] == 'x' else 0)
            prefix_dot[i+1] = prefix_dot[i] + (1 if row[i] == '.' else 0)
        for i in range(n - K + 1):
            start = i
            end = i + K
            x_count = prefix_x[end] - prefix_x[start]
            if x_count == 0:
                dot_count = prefix_dot[end] - prefix_dot[start]
                if dot_count < min_ops:
                    min_ops = dot_count
    
    # Process columns
    for j in range(W):
        col = []
        for i in range(H):
            col.append(grid[i][j])
        m = len(col)
        if K > m:
            continue
        prefix_x = [0] * (m + 1)
        prefix_dot = [0] * (m + 1)
        for i in range(m):
            prefix_x[i+1] = prefix_x[i] + (1 if col[i] == 'x' else 0)
            prefix_dot[i+1] = prefix_dot[i] + (1 if col[i] == '.' else 0)
        for i in range(m - K + 1):
            start = i
            end = i + K
            x_count = prefix_x[end] - prefix_x[start]
            if x_count == 0:
                dot_count = prefix_dot[end] - prefix_dot[start]
                if dot_count < min_ops:
                    min_ops = dot_count
    
    if min_ops == float('inf'):
        print(-1)
    else:
        print(min_ops)

if __name__ == "__main__":
    main()