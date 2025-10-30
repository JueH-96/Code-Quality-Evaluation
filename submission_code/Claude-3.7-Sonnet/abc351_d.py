from collections import deque

def has_adjacent_magnet(grid, i, j, H, W):
    """Check if a cell has an adjacent magnet."""
    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
            return True
    return False

def get_degree_of_freedom(grid, start_i, start_j, H, W):
    """Calculate the degree of freedom for a cell."""
    if grid[start_i][start_j] == '#':
        return 0
    
    visited = [[False for _ in range(W)] for _ in range(H)]
    queue = deque([(start_i, start_j)])
    visited[start_i][start_j] = True
    count = 1  # Starting cell is reachable
    
    while queue:
        i, j = queue.popleft()
        
        # If there's a magnet adjacent, can't move from this cell
        if has_adjacent_magnet(grid, i, j, H, W):
            continue
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            
            if (0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.' and 
                not visited[ni][nj]):
                visited[ni][nj] = True
                queue.append((ni, nj))
                count += 1
    
    return count

def maximum_degree_of_freedom(grid, H, W):
    """Find the maximum degree of freedom among all cells without magnets."""
    max_freedom = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                max_freedom = max(max_freedom, get_degree_of_freedom(grid, i, j, H, W))
    return max_freedom

def main():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    print(maximum_degree_of_freedom(grid, H, W))

if __name__ == "__main__":
    main()