import sys
import copy
import itertools

def generate_row_configs(first_char, N):
    configs = []
    # Determine which characters to place besides the first_char
    if first_char == 'A':
        remaining_chars = ['B', 'C']
    elif first_char == 'B':
        remaining_chars = ['A', 'C']
    else:
        remaining_chars = ['A', 'B']
    # Positions to place the remaining characters: 1 to N-1
    positions = list(range(1, N))
    # Generate all combinations of two positions
    position_combinations = list(itertools.combinations(positions, 2))
    # Generate all permutations of remaining_chars for each position combination
    for positions in position_combinations:
        for perm in itertools.permutations(remaining_chars):
            # Create the row configuration
            row = ['.' for _ in range(N)]
            row[0] = first_char
            row[positions[0]] = perm[0]
            row[positions[1]] = perm[1]
            configs.append(''.join(row))
    return configs

def is_valid_column(column, c_char, N):
    # Check if the first non-'.' character in the column matches c_char
    for char in column:
        if char != '.':
            if char != c_char:
                return False
            break
    # Check if there is exactly one 'A', one 'B', and one 'C' in the column
    count_A = column.count('A')
    count_B = column.count('B')
    count_C = column.count('C')
    # The remaining cells should be '.'s
    if count_A == 1 and count_B == 1 and count_C == 1 and len([c for c in column if c == '.']) == N - 3:
        return True
    return False

def backtrack(row, grid, C, column_chars, N):
    if row == N:
        # Check column constraints
        for col in range(N):
            column = [grid[row][col] for row in range(N)]
            if not is_valid_column(column, column_chars[col], N):
                return None
        return grid
    # Get possible row configurations based on R[row]
    row_configs = row_configs_by_row[row]
    for config in row_configs:
        grid[row] = config
        # Check column constraints incrementally
        valid = True
        for col in range(N):
            if grid[row][col] != '.':
                # Check if it's the first character in the column
                if all(grid[r][col] == '.' for r in range(row)):
                    if grid[row][col] != column_chars[col]:
                        valid = False
                        break
                # Check if the column will have more than one of any required character
                chars_in_col = [grid[r][col] for r in range(row + 1)]
                if chars_in_col.count('A') > 1 or chars_in_col.count('B') > 1 or chars_in_col.count('C') > 1:
                    valid = False
                    break
        if valid:
            result = backtrack(row + 1, grid, C, column_chars, N)
            if result is not None:
                return result
    return None

def solve(N, R, C):
    global row_configs_by_row
    row_configs_by_row = [generate_row_configs(R[i], N) for i in range(N)]
    grid = [None for _ in range(N)]
    result = backtrack(0, grid, C, list(C), N)
    if result is None:
        print("No")
    else:
        print("Yes")
        for row in result:
            print(row)

if __name__ == "__main__":
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()
    solve(N, R, C)