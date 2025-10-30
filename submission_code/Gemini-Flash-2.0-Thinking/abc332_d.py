from collections import deque

def solve():
    h, w = map(int, input().split())
    a = []
    for _ in range(h):
        a.append(list(map(int, input().split())))
    b = []
    for _ in range(h):
        b.append(list(map(int, input().split())))

    def swap_rows(grid, r):
        new_grid = [row[:] for row in grid]
        new_grid[r], new_grid[r + 1] = new_grid[r + 1], new_grid[r]
        return new_grid

    def swap_cols(grid, c):
        new_grid = [row[:] for row in grid]
        for row in new_grid:
            row[c], row[c + 1] = row[c + 1], row[c]
        return new_grid

    def are_equal(grid1, grid2):
        return all(grid1[i][j] == grid2[i][j] for i in range(h) for j in range(w))

    initial_state = tuple(tuple(row) for row in a)
    target_state = tuple(tuple(row) for row in b)

    if initial_state == target_state:
        print(0)
        return

    queue = deque([(a, 0)])
    visited = {initial_state}

    while queue:
        current_grid, current_ops = queue.popleft()

        # Try row swaps
        for i in range(h - 1):
            next_grid = swap_rows(current_grid, i)
            next_state = tuple(tuple(row) for row in next_grid)
            if next_state not in visited:
                if next_state == target_state:
                    print(current_ops + 1)
                    return
                visited.add(next_state)
                queue.append((next_grid, current_ops + 1))

        # Try column swaps
        for j in range(w - 1):
            next_grid = swap_cols(current_grid, j)
            next_state = tuple(tuple(row) for row in next_grid)
            if next_state not in visited:
                if next_state == target_state:
                    print(current_ops + 1)
                    return
                visited.add(next_state)
                queue.append((next_grid, current_ops + 1))

    print(-1)

solve()