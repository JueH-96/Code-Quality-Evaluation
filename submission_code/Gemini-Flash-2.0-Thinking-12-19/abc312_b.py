def solve():
    n, m = map(int, input().split())
    s = [input() for _ in range(n)]
    tak_code_locations = []
    for i in range(n - 8):
        for j in range(m - 8):
            is_tak_code = True
            # Condition 1: Top-left 3x3 black
            for row in range(i, i + 3):
                for col in range(j, j + 3):
                    if s[row][col] != '#':
                        is_tak_code = False
                        break
                if not is_tak_code:
                    break
            if not is_tak_code:
                continue
            # Condition 2: Bottom-right 3x3 black
            for row in range(i + 6, i + 9):
                for col in range(j + 6, j + 9):
                    if s[row][col] != '#':
                        is_tak_code = False
                        break
                if not is_tak_code:
                    break
            if not is_tak_code:
                continue
            # Condition 3: Adjacent to top-left 3x3 white
            adjacent_top_left_cells = [(i, j + 3), (i + 1, j + 3), (i + 2, j + 3), (i + 3, j), (i + 3, j + 1), (i + 3, j + 2), (i + 3, j + 3)]
            for row, col in adjacent_top_left_cells:
                if s[row][col] != '.':
                    is_tak_code = False
                    break
            if not is_tak_code:
                continue
            # Condition 4: Adjacent to bottom-right 3x3 white
            adjacent_bottom_right_cells = [(i + 6, j + 5), (i + 7, j + 5), (i + 8, j + 5), (i + 5, j + 5), (i + 5, j + 6), (i + 5, j + 7), (i + 5, j + 8)]
            for row, col in adjacent_bottom_right_cells:
                if s[row][col] != '.':
                    is_tak_code = False
                    break
            if is_tak_code:
                tak_code_locations.append((i + 1, j + 1))
                
    tak_code_locations.sort(key=lambda p: (p[0], p[1]))
    for row, col in tak_code_locations:
        print(f"{row} {col}")

if __name__ == '__main__':
    solve()