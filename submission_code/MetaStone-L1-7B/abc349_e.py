def check_line(grid):
    for i in range(3):
        if grid[i][0] == grid[i][1] == grid[i][2]:
            return True
    for j in range(3):
        if grid[0][j] == grid[1][j] == grid[2][j]:
            return True
    if grid[0][0] == grid[1][1] == grid[2][2]:
        return True
    if grid[0][2] == grid[1][1] == grid[2][0]:
        return True
    return False

def can_win(grid, is_takahashi_turn, remaining_moves):
    if remaining_moves == 0:
        return is_takahashi_turn
    for i in range(3):
        for j in range(3):
            if grid[i][j] == '.':
                new_grid = list(map(list, grid))
                new_grid[i][j] = 'R' if is_takahashi_turn else 'B'
                if check_line(new_grid):
                    return True
                new_grid[i][j] = '.'
    for i in range(3):
        for j in range(3):
            if grid[i][j] == '.' and remaining_moves > 0:
                new_grid = list(map(list, grid))
                new_grid[i][j] = 'R' if is_takahashi_turn else 'B'
                if not can_win(tuple(map(tuple, new_grid)), not is_takahashi_turn, remaining_moves - 1):
                    return True
                new_grid[i][j] = '.'
    return False

def main():
    grid = []
    for _ in range(3):
        line = input().strip()
        row = tuple(line)
        grid.append(tuple(row))
    initial_grid = tuple(tuple(row) for row in grid)
    result = can_win(initial_grid, True, 9)
    if result:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()