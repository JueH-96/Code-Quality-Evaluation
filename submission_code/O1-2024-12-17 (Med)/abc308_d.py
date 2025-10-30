def main():
    import sys
    from collections import deque
    
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    grid = input_data[2:]
    
    snuke = "snuke"
    
    # If the very first cell doesn't match 's', no valid path can start
    if grid[0][0] != 's':
        print("No")
        return
    
    # Directions for up, down, left, right
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    # visited[r][c][x] indicates whether we've visited cell (r,c)
    # when we expect the letter snuke[x].
    # x corresponds to ((t-1) mod 5) in the problem statement.
    visited = [[[False]*5 for _ in range(W)] for __ in range(H)]
    
    # Start BFS
    queue = deque()
    visited[0][0][0] = True  # We start at (0,0), which matches 's' = snuke[0]
    queue.append((0, 0, 0))
    
    while queue:
        r, c, x = queue.popleft()
        
        # If we reached the bottom-right cell, we've found a valid path
        if r == H - 1 and c == W - 1:
            print("Yes")
            return
        
        # Next index in the "snuke" cycle
        nx = (x + 1) % 5
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if not visited[nr][nc][nx] and grid[nr][nc] == snuke[nx]:
                    visited[nr][nc][nx] = True
                    queue.append((nr, nc, nx))
    
    # If BFS finishes without reaching (H-1, W-1), no valid path exists
    print("No")

# Do not forget to call main()
if __name__ == "__main__":
    main()