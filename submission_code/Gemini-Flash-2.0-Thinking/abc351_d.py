def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]

    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w

    def has_adjacent_magnet(r, c):
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and grid[nr][nc] == '#':
                return True
        return False

    max_degree_of_freedom = 0

    for start_r in range(h):
        for start_c in range(w):
            if grid[start_r][start_c] == '.':
                if has_adjacent_magnet(start_r, start_c):
                    max_degree_of_freedom = max(max_degree_of_freedom, 1)
                    continue

                reachable_cells = set()
                queue = [(start_r, start_c)]
                reachable_cells.add((start_r, start_c))

                while queue:
                    curr_r, curr_c = queue.pop(0)

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        next_r, next_c = curr_r + dr, curr_c + dc

                        if is_valid(next_r, next_c) and grid[next_r][next_c] == '.' and not has_adjacent_magnet(next_r, next_c) and (next_r, next_c) not in reachable_cells:
                            reachable_cells.add((next_r, next_c))
                            queue.append((next_r, next_c))

                max_degree_of_freedom = max(max_degree_of_freedom, len(reachable_cells))

    print(max_degree_of_freedom)

solve()