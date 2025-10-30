import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    # Parse the header: H (rows), W (columns), D (max moves)
    H, W, D = map(int, data[0].split())
    grid = data[1:1+H]
    
    # Prepare a 2D list to mark visited (humidified) floor cells.
    visited = [[False] * W for _ in range(H)]
    dq = deque()  # queue for multi-source BFS
    
    # Enqueue all humidifier cells (they are always humidified)
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                visited[i][j] = True
                dq.append((i, j, 0))  # (row index, column index, current moves)
    
    # Define moves (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Perform BFS starting from each humidifier cell
    while dq:
        r, c, d = dq.popleft()
        # If reached maximum allowed moves, we do not expand further.
        if d == D:
            continue
        nd = d + 1  # next move count
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if not visited[nr][nc] and grid[nr][nc] != '#':
                    visited[nr][nc] = True
                    dq.append((nr, nc, nd))
    
    # Count all humidified floor cells: not walls.
    result = 0
    for i in range(H):
        for j in range(W):
            if visited[i][j]:
                result += 1
    
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()