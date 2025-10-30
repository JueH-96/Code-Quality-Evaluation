def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    q_count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '?':
                q_count += 1
    
    ans = 0
    
    def is_valid(filled_grid):
        for r in range(h):
            for c in range(w):
                if r > 0 and filled_grid[r][c] == filled_grid[r-1][c]:
                    return False
                if c > 0 and filled_grid[r][c] == filled_grid[r][c-1]:
                    return False
        return True

    def fill_grid(index, current_grid):
        nonlocal ans
        if index == q_count:
            if is_valid(current_grid):
                ans = (ans + 1) % 998244353
            return

        q_index = 0
        for r in range(h):
            for c in range(w):
                if grid[r][c] == '?':
                    if q_index == index:
                        for num in range(1, 4):
                            new_grid = [list(row) for row in current_grid]
                            new_grid[r][c] = str(num)
                            fill_grid(index + 1, new_grid)
                        return
                    q_index += 1

    initial_grid = [list(row) for row in grid]
    fill_grid(0, initial_grid)
    
    print(ans)

solve()