# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import itertools
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()
    N_letters = {'A','B','C'}
    # Precompute per row the possible assignments
    row_possible_assignments = []
    for i in range(N):
        assignments = []
        # Positions to place A, B, C
        positions = [j for j in range(N)]
        for cols in itertools.combinations(positions, 3):
            for perms in itertools.permutations(['A','B','C']):
                assignment = ['.'] * N
                for idx, pos in enumerate(cols):
                    assignment[pos] = perms[idx]
                # Check leftmost character
                try:
                    leftmost_pos = assignment.index('A') if 'A' in assignment else N
                    if 'B' in assignment:
                        leftmost_pos = min(leftmost_pos, assignment.index('B'))
                    if 'C' in assignment:
                        leftmost_pos = min(leftmost_pos, assignment.index('C'))
                    if assignment[leftmost_pos] != R[i]:
                        continue  # Leftmost character does not match R[i]
                except ValueError:
                    continue  # Should not happen; we have exactly 3 letters
                assignments.append(assignment)
        row_possible_assignments.append(assignments)
    # Now proceed to assign rows recursively
    success = False
    result_grid = [''] * N
    assigned_letters_in_column = [set() for _ in range(N)]
    topmost_letter_in_column = [None for _ in range(N)]
    # For columns, keep track of which rows have been assigned
    assigned_rows_in_column = [set() for _ in range(N)]  # Not needed

    def solve(row_index):
        nonlocal success
        if success:
            return
        if row_index == N:
            # All rows assigned successfully
            success = True
            return
        for assignment in row_possible_assignments[row_index]:
            valid = True
            temp_assigned_letters_in_column = [set(s) for s in assigned_letters_in_column]
            temp_topmost_letter_in_column = topmost_letter_in_column[:]
            for j in range(N):
                ch = assignment[j]
                if ch == '.':
                    continue
                # Check column j
                if ch in temp_assigned_letters_in_column[j]:
                    valid = False
                    break
                # Update assigned letters in column j
                temp_assigned_letters_in_column[j].add(ch)
                # Check topmost character in column j
                if temp_topmost_letter_in_column[j] is None:
                    # This is the topmost character
                    if ch != C[j]:
                        valid = False
                        break
                    temp_topmost_letter_in_column[j] = ch
            if not valid:
                continue
            # Assign this assignment
            result_grid[row_index] = ''.join(assignment)
            backup_assigned_letters_in_column = [set(s) for s in assigned_letters_in_column]
            backup_topmost_letter_in_column = topmost_letter_in_column[:]
            assigned_letters_in_column = temp_assigned_letters_in_column
            topmost_letter_in_column = temp_topmost_letter_in_column
            # Proceed to next row
            solve(row_index +1)
            if success:
                return
            # Backtrack
            assigned_letters_in_column = backup_assigned_letters_in_column
            topmost_letter_in_column = backup_topmost_letter_in_column
    solve(0)
    if success:
        print("Yes")
        for row in result_grid:
            print(row)
    else:
        print("No")
threading.Thread(target=main).start()