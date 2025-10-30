import sys
from itertools import permutations

def main():
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    # Precompute configurations for each row
    configs = []
    for i in range(N):
        row_configs = []
        target_char = R[i]
        # Generate all possible (j0, j1, j2) where j0 < j1 < j2
        for j0 in range(N - 2):
            for j1 in range(j0 + 1, N - 1):
                for j2 in range(j1 + 1, N):
                    # Get the other two characters
                    other = [c for c in ['A', 'B', 'C'] if c != target_char]
                    for perm in permutations(other):
                        # Create the row
                        row = ['.'] * N
                        row[j0] = target_char
                        row[j1] = perm[0]
                        row[j2] = perm[1]
                        row_str = ''.join(row)
                        row_configs.append(row_str)
        configs.append(row_configs)

    # Now, perform backtracking
    def backtrack(current_row, columns_state, grid):
        if current_row == N:
            # Check if all columns have exactly three chars and top is correct
            for j in range(N):
                if len(columns_state[j]['chars']) != 3 or columns_state[j]['top_char'] != C[j]:
                    return None
            return grid.copy()
        # Try each configuration for current_row
        for conf in configs[current_row]:
            # Make a copy of columns_state to modify
            new_columns = []
            for col in columns_state:
                new_col = {
                    'top_row': col['top_row'],
                    'top_char': col['top_char'],
                    'chars': set(col['chars'])
                }
                new_columns.append(new_col)
            valid = True
            # Check each column in this configuration
            for j in range(N):
                c = conf[j]
                if c == '.':
                    continue
                # Process column j
                col = new_columns[j]
                if col['top_row'] == -1:
                    # First character in column j
                    if c != C[j]:
                        valid = False
                        break
                    col['top_row'] = current_row
                    col['top_char'] = c
                    col['chars'].add(c)
                else:
                    # Check if current_row is below top_row
                    if current_row <= col['top_row']:
                        valid = False
                        break
                    # Check if character is already present
                    if c in col['chars']:
                        valid = False
                        break
                    col['chars'].add(c)
                # Check column count
                if len(col['chars']) > 3:
                    valid = False
                    break
            if not valid:
                continue
            # Proceed to next row
            new_grid = grid.copy()
            new_grid.append(conf)
            result = backtrack(current_row + 1, new_columns, new_grid)
            if result is not None:
                return result
        return None

    # Initialize columns state
    initial_columns = []
    for _ in range(N):
        initial_columns.append({
            'top_row': -1,
            'top_char': None,
            'chars': set()
        })
    
    # Start backtracking from row 0
    result = backtrack(0, initial_columns, [])
    if result:
        print("Yes")
        for row in result:
            print(row)
    else:
        print("No")

if __name__ == '__main__':
    main()