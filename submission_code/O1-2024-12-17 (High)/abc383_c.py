def main():
    import sys
    from collections import deque
    
    data = sys.stdin.read().strip().split()
    H, W, D = map(int, data[:3])
    grid = data[3:]  # Each element is a string representing a row

    visited = [[False] * W for _ in range(H)]
    q = deque()

    # Multi-source BFS initialization: enqueue all humidifier cells
    idx = 0
    for r in range(H):
        row = grid[r]
        for c in range(W):
            if row[c] == 'H':
                visited[r][c] = True
                q.append((r, c, 0))  # (row, col, distance)

    # Directions for up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # BFS
    while q:
        r, c, dist = q.popleft()
        if dist < D:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if not visited[nr][nc] and grid[nr][nc] != '#':
                        visited[nr][nc] = True
                        q.append((nr, nc, dist + 1))

    # Count all visited floor cells ('.' or 'H')
    ans = 0
    for r in range(H):
        row = grid[r]
        for c in range(W):
            if visited[r][c] and row[c] != '#':
                ans += 1

    print(ans)

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()