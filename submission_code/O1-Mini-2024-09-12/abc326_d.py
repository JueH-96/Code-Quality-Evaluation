# YOUR CODE HERE
import sys
from itertools import permutations, combinations

def solve():
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    grid = [['.' for _ in range(N)] for _ in range(N)]

    # Track used letters in columns
    used_letters = [set() for _ in range(N)]
    # Track top letters in columns (row index)
    top_letters = [None for _ in range(N)]

    # Precompute all possible assignments of A, B, C to columns
    letters = ['A', 'B', 'C']
    all_assignments = []
    for cols in combinations(range(N), 3):
        for perm in permutations(letters):
            assignment = {}
            for l, c in zip(perm, cols):
                assignment[l] = c
            all_assignments.append(assignment)

    result = None

    def backtrack(row):
        nonlocal result
        if row == N:
            # All rows assigned, verify column top letters
            # Although constraints are checked during assignment
            # So we can just set the result
            result = [''.join(grid[r]) for r in range(N)]
            return True

        for assignment in all_assignments:
            # Check if letters can be placed in the columns
            valid = True
            temp_top = []
            for l in letters:
                c = assignment[l]
                if l in used_letters[c]:
                    valid = False
                    break
                if top_letters[c] is None:
                    if l != C[c]:
                        valid = False
                        break
                    temp_top.append((c, row, l))
            if not valid:
                continue
            # Determine the leftmost letter
            assigned_cols = [assignment[l] for l in letters]
            leftmost_col = min(assigned_cols)
            leftmost_letter = None
            for l in letters:
                if assignment[l] == leftmost_col:
                    leftmost_letter = l
                    break
            if leftmost_letter != R[row]:
                continue
            # Assign the letters
            for l in letters:
                c = assignment[l]
                grid[row][c] = l
                used_letters[c].add(l)
            # Update top letters
            updated = []
            for c, r, l in temp_top:
                top_letters[c] = (r, l)
                updated.append(c)
            # Recurse to next row
            if backtrack(row + 1):
                return True
            # Backtrack
            for l in letters:
                c = assignment[l]
                grid[row][c] = '.'
                used_letters[c].remove(l)
            for c in updated:
                top_letters[c] = None
        return False

    possible = backtrack(0)
    if possible and result:
        print("Yes")
        for row in result:
            print(row)
    else:
        print("No")

if __name__ == "__main__":
    solve()