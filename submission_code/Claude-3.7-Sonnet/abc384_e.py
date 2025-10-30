def max_strength(H, W, X, P, Q, grid):
    # Initialize Takahashi's strength and position (0-indexed)
    takahashi_pos = (P-1, Q-1)
    takahashi_strength = grid[takahashi_pos[0]][takahashi_pos[1]]
    
    # Mark cells as visited
    visited = [[False for _ in range(W)] for _ in range(H)]
    visited[takahashi_pos[0]][takahashi_pos[1]] = True
    
    # Frontier slimes: (strength, r, c)
    frontier = []
    
    # Directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Add adjacent slimes to the frontier
    for dr, dc in directions:
        new_r, new_c = takahashi_pos[0] + dr, takahashi_pos[1] + dc
        if 0 <= new_r < H and 0 <= new_c < W and not visited[new_r][new_c]:
            frontier.append((grid[new_r][new_c], new_r, new_c))
            visited[new_r][new_c] = True
    
    while frontier:
        # Find the slime with the highest strength that can be absorbed
        to_absorb = []
        for slime in frontier:
            s, r, c = slime
            if s * X < takahashi_strength:
                to_absorb.append(slime)
        
        if not to_absorb:
            break
        
        # Choose the slime with the highest strength
        best_slime = max(to_absorb, key=lambda x: x[0])
        frontier.remove(best_slime)
        
        # Absorb the slime
        strength, r, c = best_slime
        takahashi_strength += strength
        takahashi_pos = (r, c)
        
        # Add new adjacent slimes to the frontier
        for dr, dc in directions:
            new_r, new_c = takahashi_pos[0] + dr, takahashi_pos[1] + dc
            if 0 <= new_r < H and 0 <= new_c < W and not visited[new_r][new_c]:
                frontier.append((grid[new_r][new_c], new_r, new_c))
                visited[new_r][new_c] = True
    
    return takahashi_strength

# Read input
H, W, X = map(int, input().split())
P, Q = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(map(int, input().split())))

# Calculate and print the result
result = max_strength(H, W, X, P, Q, grid)
print(result)