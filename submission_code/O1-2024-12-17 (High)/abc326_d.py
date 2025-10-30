def main():
    import sys
    sys.setrecursionlimit(10**7)
    from itertools import combinations, permutations

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    R = input_data[1]
    C = input_data[2]

    # Grid to store our solution (initially all '.')
    grid = [['.' for _ in range(N)] for __ in range(N)]
    # How many of each letter have been used in each column
    col_count = {
        'A': [0]*N,
        'B': [0]*N,
        'C': [0]*N
    }
    # topmost[col] = index of the topmost row that has a letter in column col, or -1 if empty
    topmost = [-1]*N

    # Check feasibility of placing letter at (row, col)
    def can_place(row, col, letter):
        # We can only place this letter if the column hasn't used it yet
        if col_count[letter][col] > 0:
            return False
        # If this is the first letter in that column, it must match C[col]
        if topmost[col] == -1 and letter != C[col]:
            return False
        return True

    # Place the letter in (row, col), update data structures
    def place_letter(row, col, letter):
        grid[row][col] = letter
        col_count[letter][col] += 1
        if topmost[col] == -1:
            topmost[col] = row

    # Remove the letter from (row, col), revert data structures
    def remove_letter(row, col, letter):
        grid[row][col] = '.'
        col_count[letter][col] -= 1
        if topmost[col] == row:
            # Recompute the topmost row for this column
            new_top = -1
            for r2 in range(N):
                if grid[r2][col] != '.':
                    new_top = r2
                    break
            topmost[col] = new_top

    # Backtracking function to fill row by row
    def backtrack(row):
        # If all rows are filled, check final column constraints
        if row == N:
            # Each column must have exactly one A, one B, one C
            for cc in range(N):
                if col_count['A'][cc] != 1 or col_count['B'][cc] != 1 or col_count['C'][cc] != 1:
                    return False
            return True

        base_letter = R[row]         # Row i's leftmost letter must be R[i]
        other_letters = [x for x in "ABC" if x != base_letter]

        # We know we must place exactly one A, one B, one C in this row
        # The leftmost among the chosen columns must carry base_letter
        for c1, c2, c3 in combinations(range(N), 3):
            # We place base_letter in c1 (the smallest column), then the other two in c2, c3
            for perm in permutations(other_letters):
                l2, l3 = perm

                if (can_place(row, c1, base_letter) and
                    can_place(row, c2, l2) and
                    can_place(row, c3, l3)):
                    
                    place_letter(row, c1, base_letter)
                    place_letter(row, c2, l2)
                    place_letter(row, c3, l3)

                    if backtrack(row + 1):
                        return True

                    # revert
                    remove_letter(row, c1, base_letter)
                    remove_letter(row, c2, l2)
                    remove_letter(row, c3, l3)

        return False

    if backtrack(0):
        print("Yes")
        for row in range(N):
            print("".join(grid[row]))
    else:
        print("No")

# Do not forget to call main()
main()