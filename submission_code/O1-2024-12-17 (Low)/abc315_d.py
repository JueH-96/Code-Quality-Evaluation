def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    H = int(input_data[0])
    W = int(input_data[1])
    grid_strings = input_data[2:]

    # Convert each character to an integer color index (0..25 for 'a'..'z')
    # and store in color_idx[i][j].
    color_idx = [[0]*W for _ in range(H)]
    for i in range(H):
        row_str = grid_strings[i]
        for j, ch in enumerate(row_str):
            color_idx[i][j] = ord(ch) - ord('a')

    # row_color_count[i][c] = number of cookies of color c in row i
    row_color_count = [[0]*26 for _ in range(H)]
    # col_color_count[j][c] = number of cookies of color c in column j
    col_color_count = [[0]*26 for _ in range(W)]

    # row_count[i] = how many cookies remain in row i
    row_count = [0]*H
    # col_count[j] = how many cookies remain in column j
    col_count = [0]*W

    # alive[i][j] = True if cookie at (i,j) is still present
    alive = [[True]*W for _ in range(H)]

    # Initialize counts
    for i in range(H):
        for j in range(W):
            c = color_idx[i][j]
            row_color_count[i][c] += 1
            col_color_count[j][c] += 1
            row_count[i] += 1
            col_count[j] += 1

    # We will use a queue of (type, index), where type is 'r' or 'c'
    # to denote row or column index that needs checking.
    queue = deque()
    row_in_queue = [False]*H
    col_in_queue = [False]*W

    # Function to push a row into queue if not already
    def push_row(r):
        if not row_in_queue[r]:
            row_in_queue[r] = True
            queue.append(('r', r))

    # Function to push a column into queue if not already
    def push_col(c):
        if not col_in_queue[c]:
            col_in_queue[c] = True
            queue.append(('c', c))

    # Enqueue all rows and columns initially
    for i in range(H):
        push_row(i)
    for j in range(W):
        push_col(j)

    # BFS
    while queue:
        t, x = queue.popleft()
        if t == 'r':
            row_in_queue[x] = False
            r = x
            # Check if row r is monochromatic and has >= 2 cookies
            if row_count[r] >= 2:
                # Count how many colors > 0 and check if exactly one color
                # matches row_color_count[r][c] == row_count[r].
                # (Note: row_count[r] could be the sum of multiple colors if more than 1 color present.)
                # So effectively, we want exactly 1 color c with row_color_count[r][c] == row_count[r].
                found_color = -1
                for c in range(26):
                    if row_color_count[r][c] == row_count[r] and row_count[r] > 0:
                        found_color = c
                        break

                if found_color >= 0:
                    # This row is fully one color -> remove them all
                    # Iterate over columns to remove
                    # Each removal we decrement row_color_count[r][...] and col_color_count[col][...], etc.
                    # The total complexity remains O(H*W) over the entire run.
                    for col in range(W):
                        if alive[r][col]:
                            alive[r][col] = False
                            color_val = color_idx[r][col]
                            row_color_count[r][color_val] -= 1
                            col_color_count[col][color_val] -= 1
                            row_count[r] -= 1
                            col_count[col] -= 1
                            # Now column col might have changed, so push column col
                            push_col(col)

        else:
            col_in_queue[x] = False
            c = x
            # Similar logic for column c
            if col_count[c] >= 2:
                found_color = -1
                for color_val in range(26):
                    if col_color_count[c][color_val] == col_count[c] and col_count[c] > 0:
                        found_color = color_val
                        break

                if found_color >= 0:
                    # Remove all cookies in this column
                    for row in range(H):
                        if alive[row][c]:
                            alive[row][c] = False
                            cv = color_idx[row][c]
                            row_color_count[row][cv] -= 1
                            col_color_count[c][cv] -= 1
                            row_count[row] -= 1
                            col_count[c] -= 1
                            # This row might have changed
                            push_row(row)

    # After no more removals, count how many remain
    answer = sum(row_count)
    print(answer)

# Don't forget to call main
if __name__ == "__main__":
    main()