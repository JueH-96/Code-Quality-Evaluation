def get_adjacent_cells(i, j, H, W):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    adjacent = []
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W:
            adjacent.append((ni, nj))
    return adjacent

def solve():
    # Read input
    H, W, X = map(int, input().split())
    P, Q = map(int, input().split())
    P -= 1  # Convert to 0-based indexing
    Q -= 1
    
    # Read grid
    grid = []
    for _ in range(H):
        row = list(map(int, input().split()))
        grid.append(row)
    
    # Initialize visited set to keep track of states
    # State is (position_i, position_j, current_strength)
    from collections import deque
    visited = set()
    max_strength = grid[P][Q]
    
    # BFS queue
    queue = deque([(P, Q, grid[P][Q])])
    visited.add((P, Q, grid[P][Q]))
    
    while queue:
        curr_i, curr_j, strength = queue.popleft()
        max_strength = max(max_strength, strength)
        
        # Get adjacent cells
        for next_i, next_j in get_adjacent_cells(curr_i, curr_j, H, W):
            # Check if we can absorb this slime
            if grid[next_i][next_j] * X < strength:
                new_strength = strength + grid[next_i][next_j]
                if (next_i, next_j, new_strength) not in visited:
                    visited.add((next_i, next_j, new_strength))
                    queue.append((next_i, next_j, new_strength))
    
    print(max_strength)

# Run the solution
solve()