def solve():
    h, w, n = map(int, input().split())
    t = input()
    grid = [input() for _ in range(h)]

    possible_end_positions = set()

    for start_row in range(h):
        for start_col in range(w):
            if grid[start_row][start_col] == '.':
                current_row, current_col = start_row, start_col
                is_valid = True
                for move in t:
                    next_row, next_col = current_row, current_col
                    if move == 'L':
                        next_col -= 1
                    elif move == 'R':
                        next_col += 1
                    elif move == 'U':
                        next_row -= 1
                    elif move == 'D':
                        next_row += 1

                    if 0 <= next_row < h and 0 <= next_col < w and grid[next_row][next_col] == '.':
                        current_row, current_col = next_row, next_col
                    else:
                        is_valid = False
                        break
                if is_valid:
                    possible_end_positions.add((current_row, current_col))

    print(len(possible_end_positions))

solve()