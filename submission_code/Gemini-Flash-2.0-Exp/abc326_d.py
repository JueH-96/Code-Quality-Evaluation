def solve():
    n = int(input())
    r = input()
    c = input()

    def check(grid):
        for row in grid:
            a_count = row.count('A')
            b_count = row.count('B')
            c_count = row.count('C')
            if a_count != 1 or b_count != 1 or c_count != 1:
                return False

        for col in range(n):
            a_count = 0
            b_count = 0
            c_count = 0
            for row in range(n):
                if grid[row][col] == 'A':
                    a_count += 1
                elif grid[row][col] == 'B':
                    b_count += 1
                elif grid[row][col] == 'C':
                    c_count += 1
            if a_count != 1 or b_count != 1 or c_count != 1:
                return False

        for i in range(n):
            first_char = next((char for char in grid[i] if char != '.'), '')
            if first_char != r[i]:
                return False

        for j in range(n):
            first_char = ''
            for i in range(n):
                if grid[i][j] != '.':
                    first_char = grid[i][j]
                    break
            if first_char != c[j]:
                return False

        return True

    import itertools

    chars = ['A', 'B', 'C']
    possible_rows = []
    for perm in itertools.permutations(chars):
        possible_rows.append(list(perm))

    def find_solution():
        for row_indices in itertools.permutations(range(len(possible_rows)), n):
            grid = [['.'] * n for _ in range(n)]
            valid = True
            for i in range(n):
                first_char = possible_rows[row_indices[i]][0]
                if first_char != r[i]:
                    valid = False
                    break
                grid[i] = possible_rows[row_indices[i]].copy()

            if not valid:
                continue

            for j in range(n):
                first_char_col = grid[0][j]
                if first_char_col != c[j]:
                    
                    found_col_start = False
                    for row_perm_indices in itertools.permutations(range(len(possible_rows)), n):
                        temp_grid = [['.'] * n for _ in range(n)]
                        valid_perm = True
                        for i in range(n):
                            first_char = possible_rows[row_perm_indices[i]][0]
                            if first_char != r[i]:
                                valid_perm = False
                                break
                            temp_grid[i] = possible_rows[row_perm_indices[i]].copy()
                        
                        if not valid_perm:
                            continue
                        
                        if temp_grid[0][j] == c[j]:
                            grid = temp_grid
                            found_col_start = True
                            break
                    if not found_col_start:
                        valid = False
                        break
                
            if not valid:
                continue

            if check(grid):
                print("Yes")
                for row in grid:
                    print("".join(row))
                return

        print("No")

    find_solution()

solve()