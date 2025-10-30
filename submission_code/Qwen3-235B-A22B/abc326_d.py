import sys
import itertools
from itertools import combinations

def main():
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    # Precompute all possible row configurations
    row_configs = []
    for i in range(N):
        current_R = R[i]
        others = []
        for c in 'ABC':
            if c != current_R:
                others.append(c)
        perms = list(itertools.permutations(others))
        configs_for_row = []
        for perm in perms:
            letters = [current_R, perm[0], perm[1]]
            # Generate all combinations of 3 columns
            for cols in combinations(range(N), 3):
                filled_cells = []
                for k in range(3):
                    filled_cells.append((cols[k], letters[k]))
                # Create the row string
                row_chars = ['.'] * N
                for (col, letter) in filled_cells:
                    row_chars[col] = letter
                row_str = ''.join(row_chars)
                configs_for_row.append((filled_cells, row_str))
        row_configs.append(configs_for_row)

    # Initialize column tracking data
    first_letter = [None] * N
    count_a = [0] * N
    count_b = [0] * N
    count_c = [0] * N
    grid = []

    # Backtracking function
    def backtrack(row_idx):
        if row_idx == N:
            # Check all columns have exactly one A, B, C
            for j in range(N):
                if count_a[j] != 1 or count_b[j] != 1 or count_c[j] != 1:
                    return None
            return list(grid)
        
        for config in row_configs[row_idx]:
            filled_cells, row_str = config
            valid = True
            changes = []

            # Check each filled cell for validity
            for (j, letter) in filled_cells:
                if letter == 'A' and count_a[j] > 0:
                    valid = False
                    break
                if letter == 'B' and count_b[j] > 0:
                    valid = False
                    break
                if letter == 'C' and count_c[j] > 0:
                    valid = False
                    break
                if first_letter[j] is None and letter != C[j]:
                    valid = False
                    break

            if not valid:
                continue

            saved = []
            valid_apply = True
            # Apply changes and check validity
            for (j, letter) in filled_cells:
                prev_first = first_letter[j]
                prev_a, prev_b, prev_c = count_a[j], count_b[j], count_c[j]
                saved.append((j, prev_first, prev_a, prev_b, prev_c))

                if letter == 'A':
                    count_a[j] += 1
                elif letter == 'B':
                    count_b[j] += 1
                else:
                    count_c[j] += 1

                if first_letter[j] is None:
                    first_letter[j] = letter

                if (count_a[j] > 1 or count_b[j] > 1 or count_c[j] > 1):
                    valid_apply = False
                    break

            if not valid_apply:
                # Undo saved changes
                for (j, prev_first, prev_a, prev_b, prev_c) in saved:
                    first_letter[j] = prev_first
                    count_a[j], count_b[j], count_c[j] = prev_a, prev_b, prev_c
                continue

            # Proceed to next row
            grid.append(row_str)
            result = backtrack(row_idx + 1)
            if result is not None:
                return result
            grid.pop()

            # Undo changes
            for (j, prev_first, prev_a, prev_b, prev_c) in saved:
                first_letter[j] = prev_first
                count_a[j], count_b[j], count_c[j] = prev_a, prev_b, prev_c

        return None

    solution = backtrack(0)
    if solution:
        print("Yes")
        for row in solution:
            print(row)
    else:
        print("No")

if __name__ == "__main__":
    main()