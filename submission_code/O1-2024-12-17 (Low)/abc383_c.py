def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    H, W, D = map(int, input_data[:3])
    grid_lines = input_data[3:]
    
    # Store the grid in a list of strings
    grid = grid_lines
    
    # Directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # A 2D array to record the distance from the nearest humidifier.
    # Initialize with -1 (meaning unvisited).
    dist = [[-1]*W for _ in range(H)]
    
    # Multi-source BFS initialization
    queue = deque()
    
    # Enqueue all humidifier cells, mark their distance as 0
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'H':
                dist[r][c] = 0
                queue.append((r, c))
    
    # Perform BFS
    while queue:
        r, c = queue.popleft()
        current_dist = dist[r][c]
        
        # If we've reached the maximum distance D, don't spread further
        if current_dist == D:
            continue
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Check boundaries and skip walls
            if 0 <= nr < H and 0 <= nc < W:
                if grid[nr][nc] != '#':  # not a wall
                    # If unvisited, visit with distance+1
                    if dist[nr][nc] == -1:
                        dist[nr][nc] = current_dist + 1
                        queue.append((nr, nc))
    
    # Count all floor cells ('.') or humidifier cells ('H') that are visited
    answer = 0
    for r in range(H):
        for c in range(W):
            if grid[r][c] != '#':  # floor or humidifier
                if dist[r][c] != -1:  # visited (humidified)
                    answer += 1
    
    print(answer)

# Do not forget to call main()!
if __name__ == "__main__":
    main()