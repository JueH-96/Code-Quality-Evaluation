from collections import deque

def solve():
    h, w = map(int, input().split())
    a = []
    for _ in range(h):
        a.append(list(map(int, input().split())))
    b = []
    for _ in range(h):
        b.append(list(map(int, input().split())))

    def get_grid_tuple(grid):
        return tuple(tuple(row) for row in grid)

    def swap_rows(grid, i):
        new_grid = [row[:] for row in grid]
        new_grid[i], new_grid[i+1] = new_grid[i+1], new_grid[i]
        return new_grid

    def swap_cols(grid, j):
        new_grid = [row[:] for row in grid]
        for row in new_grid:
            row[j], row[j+1] = row[j+1], row[j]
        return new_grid

    q = deque()
    q.append((a, 0))
    visited = {get_grid_tuple(a)}

    while q:
        curr_grid, steps = q.popleft()
        if curr_grid == b:
            print(steps)
            return

        for i in range(h - 1):
            next_grid = swap_rows(curr_grid, i)
            grid_tuple = get_grid_tuple(next_grid)
            if grid_tuple not in visited:
                visited.add(grid_tuple)
                q.append((next_grid, steps + 1))

        for j in range(w - 1):
            next_grid = swap_cols(curr_grid, j)
            grid_tuple = get_grid_tuple(next_grid)
            if grid_tuple not in visited:
                visited.add(grid_tuple)
                q.append((next_grid, steps + 1))
    
    print("-1")

solve()