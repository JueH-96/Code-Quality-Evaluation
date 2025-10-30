def solve():
    n, m = map(int, input().split())
    grid_str = [input() for _ in range(n)]
    grid = [list(row) for row in grid_str]

    start_row, start_col = 1, 1

    reachable = set([(start_row, start_col)])
    queue = [(start_row, start_col)]

    while queue:
        r, c = queue.pop(0)

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            move_r, move_c = r, c
            while 0 <= move_r + dr < n and 0 <= move_c + dc < m and grid[move_r + dr][move_c + dc] == '.':
                move_r += dr
                move_c += dc
                if (move_r, move_c) not in reachable:
                    reachable.add((move_r, move_c))
                    queue.append((move_r, move_c))

            final_r, final_c = r, c
            while 0 <= final_r + dr < n and 0 <= final_c + dc < m and grid[final_r + dr][final_c + dc] == '.':
                final_r += dr
                final_c += dc

            curr_r, curr_c = r, c
            while (curr_r, curr_c) != (final_r, final_c):
                curr_r += dr
                curr_c += dc
                if 0 <= curr_r < n and 0 <= curr_c < m and grid[curr_r][curr_c] == '.' and (curr_r, curr_c) not in reachable:
                    reachable.add((curr_r, curr_c))
                    queue.append((curr_r, curr_c))

    print(len(reachable))

solve()