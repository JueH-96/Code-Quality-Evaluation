def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]

    for r_eaten in range(1, h + 1):
        for c_eaten in range(1, w + 1):
            if grid[r_eaten - 1][c_eaten - 1] == '.':
                potential_grid = [list(row) for row in grid]
                potential_grid[r_eaten - 1] = potential_grid[r_eaten - 1][:c_eaten - 1] + '#' + potential_grid[r_eaten - 1][c_eaten:]

                min_r, max_r, min_c, max_c = None, None, None, None

                for r in range(h):
                    for c in range(w):
                        if potential_grid[r][c] == '#':
                            if min_r is None:
                                min_r, max_r = r + 1, r + 1
                                min_c, max_c = c + 1, c + 1
                            else:
                                min_r = min(min_r, r + 1)
                                max_r = max(max_r, r + 1)
                                min_c = min(min_c, c + 1)
                                max_c = max(max_c, c + 1)

                if min_r is not None:
                    if max_r > min_r and max_c > min_c:
                        is_rectangle = True
                        for r in range(min_r - 1, max_r):
                            for c in range(min_c - 1, max_c):
                                if potential_grid[r][c] != '#':
                                    is_rectangle = False
                                    break
                            if not is_rectangle:
                                break
                        if is_rectangle:
                            print(r_eaten, c_eaten)
                            return

solve()