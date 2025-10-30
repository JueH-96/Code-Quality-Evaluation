def can_move(H, W, grid, r, c):
    # Check if position is valid and has no adjacent magnets
    if r < 0 or r >= H or c < 0 or c >= W or grid[r][c] == '#':
        return False
        
    # Check adjacent cells for magnets
    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '#':
            return False
    return True

def count_reachable(H, W, grid, start_r, start_c):
    if not can_move(H, W, grid, start_r, start_c):
        return 1 if grid[start_r][start_c] == '.' else 0
        
    visited = set()
    visited.add((start_r, start_c))
    
    stack = [(start_r, start_c)]
    while stack:
        r, c = stack.pop()
        
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and can_move(H, W, grid, nr, nc):
                visited.add((nr, nc))
                stack.append((nr, nc))
                
    return len(visited)

# Read input
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())

# Find maximum degree of freedom
max_freedom = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            freedom = count_reachable(H, W, grid, i, j)
            max_freedom = max(max_freedom, freedom)

print(max_freedom)