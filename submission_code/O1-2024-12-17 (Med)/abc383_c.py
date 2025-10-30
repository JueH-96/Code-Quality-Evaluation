from collections import deque

def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, D = map(int, input_data[:3])
    grid_lines = input_data[3:]
    
    # Read the grid
    grid = grid_lines
    
    # Prepare distance array, -1 means unvisited
    dist = [[-1]*W for _ in range(H)]
    
    # Multi-source BFS initialization
    q = deque()
    # Find humidifier cells
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                dist[i][j] = 0
                q.append((i, j))
    
    # Directions for up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS
    while q:
        r, c = q.popleft()
        # If we've reached distance D, don't expand further
        if dist[r][c] == D:
            continue
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if grid[nr][nc] != '#' and dist[nr][nc] == -1:
                    dist[nr][nc] = dist[r][c] + 1
                    q.append((nr, nc))
    
    # Count how many cells (floor '.' or humidifier 'H') have dist != -1
    # (meaning they are humidified)
    answer = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '#' and dist[i][j] != -1:
                answer += 1
    
    print(answer)

# Do not forget to call main
if __name__ == "__main__":
    main()