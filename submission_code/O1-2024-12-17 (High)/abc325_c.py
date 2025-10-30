def main():
    import sys
    from collections import deque
    
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    grid = input_data[2:]
    
    # Mark sensor cells
    sensors = [[(grid[r][c] == '#') for c in range(W)] for r in range(H)]
    
    visited = [[False]*W for _ in range(H)]
    
    # Directions: 8-adjacency (horizontal, vertical, diagonal)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),           ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]
    
    def bfs(sr, sc):
        queue = deque([(sr, sc)])
        visited[sr][sc] = True
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if sensors[nr][nc] and not visited[nr][nc]:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
    
    # Count connected components of sensors
    count = 0
    for i in range(H):
        for j in range(W):
            if sensors[i][j] and not visited[i][j]:
                # Found a new cluster of sensors, BFS/DFS over it
                bfs(i, j)
                count += 1
    
    print(count)

# Do not forget to call main
main()