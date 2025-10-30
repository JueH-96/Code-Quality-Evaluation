# YOUR CODE HERE
import itertools

def solve():
    H, W = map(int, input().split())
    A = []
    for _ in range(H):
        A.append(list(map(int, input().split())))
    B = []
    for _ in range(H):
        B.append(list(map(int, input().split())))

    def grid_to_tuple(grid):
        return tuple(tuple(row) for row in grid)

    q = [(A, 0)]
    visited = {grid_to_tuple(A)}
    
    while q:
        curr_grid, dist = q.pop(0)
        if curr_grid == B:
            print(dist)
            return

        if dist > 20:
            continue
        
        # Row swaps
        for i in range(H - 1):
            next_grid = [list(row) for row in curr_grid]
            next_grid[i], next_grid[i+1] = next_grid[i+1], next_grid[i]
            
            grid_tuple = grid_to_tuple(next_grid)
            if grid_tuple not in visited:
                visited.add(grid_tuple)
                q.append((next_grid, dist + 1))

        # Column swaps
        for j in range(W - 1):
            next_grid = [list(row) for row in curr_grid]
            for i in range(H):
                next_grid[i][j], next_grid[i][j+1] = next_grid[i][j+1], next_grid[i][j]
            
            grid_tuple = grid_to_tuple(next_grid)
            if grid_tuple not in visited:
                visited.add(grid_tuple)
                q.append((next_grid, dist + 1))
    
    print(-1)

solve()