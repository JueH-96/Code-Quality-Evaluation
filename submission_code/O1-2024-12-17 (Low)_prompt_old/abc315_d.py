def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast parsing
    H, W = map(int, input_data[:2])
    cookies = input_data[2:]  # list of H strings

    # We will keep track of which rows/columns are still "active" (not removed).
    # A cookie at (r,c) is present iff row r is active AND column c is active.
    # When a row is found to be uniform (all its active cookies share the same color, and at least 2 cookies remain),
    # we remove that row. Similarly for columns. We repeat until no row/column can be removed.
    # In the end, the number of remaining cookies is (number_of_active_rows) * (number_of_active_columns).

    # For efficiency, we maintain:
    # rowFreq[r][color] = how many active columns in row r have this color
    # colFreq[c][color] = how many active rows in column c have this color
    # rowSize[r] = how many active columns remain in row r
    # colSize[c] = how many active rows remain in column c
    # rowActive[r], colActive[c] = boolean indicating if row r or column c is still active
    #
    # We use a queue to repeatedly remove rows/columns as soon as they become uniform.

    from collections import deque

    # Initialize data structures
    rowActive = [True]*H
    colActive = [True]*W
    rowSize = [0]*H
    colSize = [0]*W

    # Frequencies for rows and columns (26 letters)
    rowFreq = [[0]*26 for _ in range(H)]
    colFreq = [[0]*26 for _ in range(W)]

    # Fill in rowFreq, colFreq, rowSize, colSize
    # Initially, all rows and all columns are active
    for r in range(H):
        for c in range(W):
            color_index = ord(cookies[r][c]) - ord('a')
            rowFreq[r][color_index] += 1
            colFreq[c][color_index] += 1
        rowSize[r] = W  # initially, each row has W active columns
    for c in range(W):
        colSize[c] = H  # initially, each column has H active rows

    # Function to check if a row is uniform (and has at least 2 active cookies)
    def is_uniform_row(r):
        if rowSize[r] < 2:
            return False
        rs = rowSize[r]
        freq = rowFreq[r]
        # Check if there's a color with frequency == rowSize[r]
        # Since we only have 26 colors, we can just loop
        for cnt in freq:
            if cnt == rs:
                return True
        return False

    # Function to check if a column is uniform (and has at least 2 active cookies)
    def is_uniform_col(c):
        if colSize[c] < 2:
            return False
        cs = colSize[c]
        freq = colFreq[c]
        # Check if there's a color with frequency == colSize[c]
        for cnt in freq:
            if cnt == cs:
                return True
        return False

    # Build initial queue
    Q = deque()
    for r in range(H):
        if rowSize[r] >= 2 and is_uniform_row(r):
            Q.append(('r', r))
    for c in range(W):
        if colSize[c] >= 2 and is_uniform_col(c):
            Q.append(('c', c))

    # Active row/col counts
    active_row_count = H
    active_col_count = W

    while Q:
        typ, idx = Q.popleft()
        if typ == 'r':
            r = idx
            if not rowActive[r]:
                continue
            # Re-check if it's still uniform
            if rowSize[r] >= 2 and is_uniform_row(r):
                # Remove row r
                rowActive[r] = False
                active_row_count -= 1
                # Update columns
                # For each column c that is still active, reduce freq
                # O(W) per removed row
                for c in range(W):
                    if colActive[c]:
                        color_index = ord(cookies[r][c]) - ord('a')
                        colFreq[c][color_index] -= 1
                        colSize[c] -= 1
                        # Check if col became uniform
                        if colSize[c] >= 2 and is_uniform_col(c):
                            Q.append(('c', c))

        else:  # typ == 'c'
            c = idx
            if not colActive[c]:
                continue
            # Re-check if it's still uniform
            if colSize[c] >= 2 and is_uniform_col(c):
                # Remove column c
                colActive[c] = False
                active_col_count -= 1
                # Update rows
                # For each row r that is still active, reduce freq
                # O(H) per removed column
                for r in range(H):
                    if rowActive[r]:
                        color_index = ord(cookies[r][c]) - ord('a')
                        rowFreq[r][color_index] -= 1
                        rowSize[r] -= 1
                        # Check if row became uniform
                        if rowSize[r] >= 2 and is_uniform_row(r):
                            Q.append(('r', r))

    # Final number of cookies = number_of_active_rows * number_of_active_columns
    print(active_row_count * active_col_count)

def main():
    solve()

# Call solve() as requested
if __name__ == "__main__":
    main()