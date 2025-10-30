def solve():
    h, w = map(int, input().split())
    si, sj = map(int, input().split())
    grid = [input() for _ in range(h)]
    actions = input()

    current_row = si
    current_col = sj

    for action in actions:
        new_row, new_col = current_row, current_col
        if action == 'L':
            if current_col > 1 and grid[current_row - 1][current_col - 2] == '.':
                new_col -= 1
        elif action == 'R':
            if current_col < w and grid[current_row - 1][current_col] == '.':
                new_col += 1
        elif action == 'U':
            if current_row > 1 and grid[current_row - 2][current_col - 1] == '.':
                new_row -= 1
        elif action == 'D':
            if current_row < h and grid[current_row][current_col - 1] == '.':
                new_row += 1
        current_row, current_col = new_row, new_col

    print(current_row, current_col)

solve()