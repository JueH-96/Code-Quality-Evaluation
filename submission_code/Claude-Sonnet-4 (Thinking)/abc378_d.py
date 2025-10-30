H, W, K = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find all empty cells and create a mapping
empty_cells = []
pos_to_empty_idx = {}
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            pos_to_empty_idx[(i, j)] = len(empty_cells)
            empty_cells.append((i, j))

def is_valid(i, j):
    return 0 <= i < H and 0 <= j < W and grid[i][j] == '.'

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

memo = {}

def dfs(i, j, moves_left, visited_mask):
    if moves_left == 0:
        return 1
    
    state = (i, j, moves_left, visited_mask)
    if state in memo:
        return memo[state]
    
    result = 0
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if is_valid(ni, nj):
            new_idx = pos_to_empty_idx[(ni, nj)]
            if not (visited_mask & (1 << new_idx)):
                new_visited = visited_mask | (1 << new_idx)
                result += dfs(ni, nj, moves_left - 1, new_visited)
    
    memo[state] = result
    return result

total = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            idx = pos_to_empty_idx[(i, j)]
            visited_mask = 1 << idx
            total += dfs(i, j, K, visited_mask)

print(total)