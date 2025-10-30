def main():
    import sys
    sys.setrecursionlimit(10**7)
    from itertools import combinations, permutations

    # Read inputs
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()

    # Grid to store the final placement ('.' means empty)
    grid = [['.'] * N for _ in range(N)]
    # For each column j, track how many times we've placed A, B, C
    colCount = [ {'A':0, 'B':0, 'C':0} for _ in range(N) ]
    # topUsed[j] = False means column j has not yet placed its topmost letter,
    # so the first time we place a letter in that column it must match C[j].
    topUsed = [False]*N

    # Backtracking function
    def backtrack(row):
        # If we've assigned letters for all rows, check column usage validity
        if row == N:
            # Each column must have exactly one A, one B, one C
            for j in range(N):
                for letter in 'ABC':
                    if colCount[j][letter] != 1:
                        return False
            return True

        # We must place exactly {A, B, C} in row "row", with the leftmost letter = R[row].
        leftmost_letter = R[row]
        remaining_letters = {'A','B','C'}
        remaining_letters.remove(leftmost_letter)
        remaining_letters = list(remaining_letters)  # 2 letters left

        # Choose exactly 3 distinct columns (col1 < col2 < col3)
        for cols in combinations(range(N), 3):
            col1, col2, col3 = cols  # they are in ascending order
            
            # We will try to place:
            #   leftmost_letter in col1,
            #   then the other two letters in col2, col3 in some permutation.
            for perm_two in permutations(remaining_letters):
                changes = []  # keep track of all placements so we can revert if needed

                # Attempt to place letter "let" in (row, col). Returns True if successful.
                def placeLetter(r, c, let):
                    # If column c is unused so far, it must match the topmost letter constraint:
                    if not topUsed[c]:
                        if let != C[c]:
                            return False
                    # If we already used this letter in column c, can't use it again
                    if colCount[c][let] == 1:
                        return False

                    # Record old state for undo
                    old_grid_val = grid[r][c]
                    changes.append(('grid', r, c, old_grid_val))
                    grid[r][c] = let

                    # Increase count for that letter in the column
                    changes.append(('colCount', c, let))
                    colCount[c][let] += 1

                    # If this is the first time using column c, mark topUsed[c]
                    if not topUsed[c]:
                        changes.append(('topUsed', c))
                        topUsed[c] = True
                    return True

                # Revert all changes made in this step
                def revert(changes_list):
                    while changes_list:
                        item = changes_list.pop()
                        if item[0] == 'grid':
                            # Restore grid cell
                            _, r0, c0, old_val = item
                            grid[r0][c0] = old_val
                        elif item[0] == 'colCount':
                            # Decrement the column count
                            _, c0, let0 = item
                            colCount[c0][let0] -= 1
                        elif item[0] == 'topUsed':
                            # Mark column as unused again
                            _, c0 = item
                            topUsed[c0] = False

                # Now place:
                #   R[row] in col1
                #   perm_two[0] in col2
                #   perm_two[1] in col3
                if not placeLetter(row, col1, leftmost_letter):
                    revert(changes)
                    continue
                if not placeLetter(row, col2, perm_two[0]):
                    revert(changes)
                    continue
                if not placeLetter(row, col3, perm_two[1]):
                    revert(changes)
                    continue

                # If we've placed all three letters, move to the next row
                if backtrack(row + 1):
                    return True
                # Otherwise revert and try another arrangement
                revert(changes)

        # If no column combination or permutation works for this row, fail
        return False

    # Run backtracking from row 0
    if backtrack(0):
        print("Yes")
        for i in range(N):
            print("".join(grid[i]))
    else:
        print("No")

# Don't forget to call main()
if __name__ == "__main__":
    main()