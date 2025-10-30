def solve():
    n = int(input())
    r = input()
    c = input()

    def is_valid(grid):
        for i in range(n):
            row_chars = [cell for cell in grid[i] if cell != '.']
            if len(row_chars) != len(set(row_chars)):
                return False
            if 'A' not in row_chars or 'B' not in row_chars or 'C' not in row_chars:
                return False

        for j in range(n):
            col_chars = [grid[i][j] for i in range(n) if grid[i][j] != '.']
            if len(col_chars) != len(set(col_chars)):
                return False
            if 'A' not in col_chars or 'B' not in col_chars or 'C' not in col_chars:
                return False
        return True

    def check_leftmost(grid):
        for i in range(n):
            first_char = None
            for char in grid[i]:
                if char != '.':
                    first_char = char
                    break
            if first_char is not None and first_char != r[i]:
                return False
            elif first_char is None and r[i] != '.': # Should not happen if grid is partially filled correctly
                return False
        return True

    def check_topmost(grid):
        for j in range(n):
            first_char = None
            for i in range(n):
                if grid[i][j] != '.':
                    first_char = grid[i][j]
                    break
            if first_char is not None and first_char != c[j]:
                return False
            elif first_char is None and c[j] != '.': # Should not happen if grid is partially filled correctly
                return False
        return True

    def find_empty(grid):
        for i in range(n):
            for j in range(n):
                if grid[i][j] == '.':
                    return i, j
        return -1, -1

    def backtrack(grid):
        row, col = find_empty(grid)
        if row == -1:
            if is_valid(grid) and check_leftmost(grid) and check_topmost(grid):
                return ["".join(row) for row in grid]
            else:
                return None

        for char in 'ABC':
            can_place_row = grid[row].count(char) == 0
            can_place_col = [grid[i][col] for i in range(n)].count(char) == 0

            if can_place_row and can_place_col:
                new_grid = [list(r) for r in grid]
                new_grid[row][col] = char

                is_potential_leftmost = True
                first_in_row = None
                for k in range(n):
                    if new_grid[row][k] != '.':
                        first_in_row = new_grid[row][k]
                        break
                if first_in_row is not None and first_in_row != r[row]:
                    is_potential_leftmost = False

                is_potential_topmost = True
                first_in_col = None
                for k in range(n):
                    if new_grid[k][col] != '.':
                        first_in_col = new_grid[k][col]
                        break
                if first_in_col is not None and first_in_col != c[col]:
                    is_potential_topmost = False

                if is_potential_leftmost and is_potential_topmost:
                    result = backtrack(new_grid)
                    if result:
                        return result
        return None

    initial_grid = [['.' for _ in range(n)] for _ in range(n)]

    result = backtrack(initial_grid)

    if result:
        print("Yes")
        for row_str in result:
            print(row_str)
    else:
        print("No")

solve()