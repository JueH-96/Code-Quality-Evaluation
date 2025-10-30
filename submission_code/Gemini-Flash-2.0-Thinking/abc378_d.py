def solve():
    h, w, k = map(int, input().split())
    grid = [input() for _ in range(h)]
    count = 0

    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w and grid[r][c] == '.'

    def dfs(r, c, moves_left, visited):
        nonlocal count
        if moves_left == 0:
            count += 1
            return

        if not is_valid(r, c) or (r, c) in visited:
            return

        visited.add((r, c))

        # Explore adjacent cells
        dfs(r + 1, c, moves_left - 1, visited)
        dfs(r - 1, c, moves_left - 1, visited)
        dfs(r, c + 1, moves_left - 1, visited)
        dfs(r, c - 1, moves_left - 1, visited)

        visited.remove((r, c))  # Backtrack

    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                dfs(r, c, k, set())

    print(count)

solve()