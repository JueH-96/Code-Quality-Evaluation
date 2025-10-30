def solve():
    n = int(input())
    grid = [[0 for _ in range(n)] for _ in range(n)]
    center_row = (n - 1) // 2
    center_col = (n - 1) // 2
    grid[center_row][center_col] = 'T'

    current_part = 1
    start_row = center_row - 1
    start_col = center_col

    if 0 <= start_row < n and 0 <= start_col < n:
        grid[start_row][start_col] = current_part
        current_part += 1
        current_row = start_row
        current_col = start_col
    else:
        start_row = center_row
        start_col = center_col - 1
        grid[start_row][start_col] = current_part
        current_part += 1
        current_row = start_row
        current_col = start_col

    def is_valid_empty(r, c):
        return 0 <= r < n and 0 <= c < n and grid[r][c] == 0

    while current_part <= n * n - 1:
        moved = False
        moves = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        for dr, dc in moves:
            next_r = current_row + dr
            next_c = current_col + dc
            if is_valid_empty(next_r, next_c):
                grid[next_r][next_c] = current_part
                current_row = next_r
                current_col = next_c
                current_part += 1
                moved = True
                break
        if not moved:
            # This should not happen based on the problem guarantee
            break

    for row in grid:
        print(*row)

solve()