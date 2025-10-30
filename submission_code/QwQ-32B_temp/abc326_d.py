import sys
import itertools

def main():
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    if len(R) != N or len(C) != N:
        print("No")
        return

    def backtrack(row, started, first_char, chars_in_col, grid):
        if row == N:
            # Check all columns have exactly 3 letters
            for j in range(N):
                if len(chars_in_col[j]) != 3:
                    return None
            # Check first_char matches C
            for j in range(N):
                if first_char[j] != C[j]:
                    return None
            return grid
        for cols in itertools.combinations(range(N), 3):
            min_col = min(cols)
            target = R[row]
            remaining_letters = ['A', 'B', 'C']
            remaining_letters.remove(target)
            for perm in itertools.permutations(remaining_letters):
                # Assign letters to columns
                letter_assignment = {}
                letter_assignment[min_col] = target
                other_cols = []
                for col in cols:
                    if col != min_col:
                        other_cols.append(col)
                letter_assignment[other_cols[0]] = perm[0]
                letter_assignment[other_cols[1]] = perm[1]
                valid = True
                for col in cols:
                    letter = letter_assignment[col]
                    if not started[col]:
                        if letter != C[col]:
                            valid = False
                            break
                    if letter in chars_in_col[col]:
                        valid = False
                        break
                if not valid:
                    continue
                # Create new states
                new_started = started.copy()
                new_first_char = first_char.copy()
                new_chars_in_col = [s.copy() for s in chars_in_col]
                new_grid = [list(r) for r in grid]
                new_row = ['.'] * N
                for col in cols:
                    new_row[col] = letter_assignment[col]
                new_grid[row] = new_row
                # Update states
                for col in cols:
                    letter = letter_assignment[col]
                    if not started[col]:
                        new_started[col] = True
                        new_first_char[col] = letter
                    new_chars_in_col[col].add(letter)
                result = backtrack(row + 1, new_started, new_first_char, new_chars_in_col, new_grid)
                if result is not None:
                    return result
        return None

    started = [False] * N
    first_char = [None] * N
    chars_in_col = [set() for _ in range(N)]
    grid = [['.'] * N for _ in range(N)]

    result = backtrack(0, started, first_char, chars_in_col, grid)
    if result is not None:
        print("Yes")
        for row in result:
            print(''.join(row))
    else:
        print("No")

if __name__ == '__main__':
    main()