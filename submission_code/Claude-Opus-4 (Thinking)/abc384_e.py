H, W, X = map(int, input().split())
P, Q = map(int, input().split())
P -= 1  # Convert to 0-indexed
Q -= 1

grid = []
for _ in range(H):
    row = list(map(int, input().split()))
    grid.append(row)

# Directions for adjacent cells (up, down, left, right)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

max_strength = grid[P][Q]

def dfs(absorbed_set, current_pos, strength):
    global max_strength
    max_strength = max(max_strength, strength)
    
    row, col = current_pos
    
    # Try to absorb adjacent slimes
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in absorbed_set:
            if grid[nr][nc] < strength / X:
                # Absorb this slime
                new_absorbed = absorbed_set | {(nr, nc)}
                dfs(new_absorbed, (nr, nc), strength + grid[nr][nc])

initial_absorbed = {(P, Q)}
dfs(initial_absorbed, (P, Q), grid[P][Q])
print(max_strength)