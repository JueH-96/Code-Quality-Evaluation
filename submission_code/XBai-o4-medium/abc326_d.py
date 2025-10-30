import sys
import itertools

def main():
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    # Generate all possible row configurations for each row
    rows_options = [[] for _ in range(N)]
    for i in range(N):
        R_char = R[i]
        remaining = list(set("ABC") - {R_char})
        perms = list(itertools.permutations(remaining))
        possible_perms = [ [R_char] + list(p) for p in perms ]
        column_combos = list(itertools.combinations(range(N), 3))
        for perm in possible_perms:
            for cols in column_combos:
                row = ['.'] * N
                row[cols[0]] = perm[0]
                row[cols[1]] = perm[1]
                row[cols[2]] = perm[2]
                row_str = ''.join(row)
                rows_options[i].append(row_str)

    # Backtracking function to find a valid grid configuration
    def backtrack(row_index, columns_first, columns_counts, grid_so_far):
        if row_index == N:
            # Check if all columns have exactly 3 letters (A, B, C)
            for j in range(N):
                total = sum(columns_counts[j][c] for c in 'ABC')
                if total != 3:
                    return None
            return grid_so_far
        # Try each possible configuration for the current row
        for config in rows_options[row_index]:
            # Precompute positions with non '.' characters
            positions = []
            for j in range(N):
                if config[j] != '.':
                    positions.append( (j, config[j]) )
            # Check validity of this configuration
            valid = True
            for (j, c) in positions:
                # Check column first character
                if columns_first[j] is None:
                    if c != C[j]:
                        valid = False
                        break
                # Check column counts
                if columns_counts[j][c] >= 1:
                    valid = False
                    break
            if not valid:
                continue
            # Prepare new state for recursion
            new_columns_first = columns_first.copy()
            new_columns_counts = [ {k: v for k, v in cnt.items()} for cnt in columns_counts ]
            new_grid = grid_so_far + [config]
            # Update new_columns_first and new_columns_counts
            for (j, c) in positions:
                if new_columns_first[j] is None:
                    new_columns_first[j] = c
                new_columns_counts[j][c] += 1
            # Recurse
            result = backtrack(row_index + 1, new_columns_first, new_columns_counts, new_grid)
            if result is not None:
                return result
        return None

    # Initial state for backtracking
    initial_columns_first = [None] * N
    initial_columns_counts = [ {'A':0, 'B':0, 'C':0} for _ in range(N) ]
    solution = backtrack(0, initial_columns_first, initial_columns_counts, [])

    if solution is not None:
        print("Yes")
        for row in solution:
            print(row)
    else:
        print("No")

if __name__ == '__main__':
    main()