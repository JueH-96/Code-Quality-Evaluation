def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read inputs
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    K = int(next(it))
    grid = [next(it) for _ in range(H)]
    
    INF = 10**9
    ans = INF
    
    # Check horizontal segments:
    for r in range(H):
        row = grid[r]
        # Build prefix sums for x and dot for this row.
        prefix_x = [0] * (W + 1)
        prefix_dot = [0] * (W + 1)
        for j in range(W):
            prefix_x[j+1] = prefix_x[j] + (1 if row[j] == 'x' else 0)
            prefix_dot[j+1] = prefix_dot[j] + (1 if row[j] == '.' else 0)
        # Sliding window for segments of length K.
        for j in range(W - K + 1):
            # If there's no x in the segment, it's possible to turn '.' into o.
            if prefix_x[j+K] - prefix_x[j] == 0:
                cost = prefix_dot[j+K] - prefix_dot[j]
                ans = min(ans, cost)
    
    # Check vertical segments:
    for c in range(W):
        # Build prefix sums for column c.
        prefix_x = [0] * (H + 1)
        prefix_dot = [0] * (H + 1)
        for r in range(H):
            cell = grid[r][c]
            prefix_x[r+1] = prefix_x[r] + (1 if cell == 'x' else 0)
            prefix_dot[r+1] = prefix_dot[r] + (1 if cell == '.' else 0)
        # Slide through vertical segments of length K.
        for r in range(H - K + 1):
            if prefix_x[r+K] - prefix_x[r] == 0:
                cost = prefix_dot[r+K] - prefix_dot[r]
                ans = min(ans, cost)
    
    # If no valid segment could be made, output -1.
    if ans == INF:
        sys.stdout.write("-1")
    else:
        sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()