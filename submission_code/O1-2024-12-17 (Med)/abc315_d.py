def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    cookies = input_data[2:]

    # Convert each row-string into a list (or just refer to cookies[i][j])
    # We'll treat cookies[i] as the i-th row (0-based).
    # cookies[i][j] is a character in 'a'..'z'.

    # Precompute the color index for every (i, j).
    # color_index[i][j] = ord(cookies[i][j]) - ord('a')
    # However, to save memory, we'll just compute on the fly when needed.

    # We maintain:
    #  - row_removed[i]: whether row i is fully removed
    #  - col_removed[j]: whether column j is fully removed
    #  - row_count[i]: how many cookies are still active in row i
    #  - col_count[j]: how many cookies are still active in column j
    #  - row_color_counts[i][0..25]: how many cookies of each color remain in row i
    #  - col_color_counts[j][0..25]: how many cookies of each color remain in column j

    row_removed = [False]*H
    col_removed = [False]*W
    row_count = [0]*H
    col_count = [0]*W

    # Initialize color counts
    row_color_counts = [[0]*26 for _ in range(H)]
    col_color_counts = [[0]*26 for _ in range(W)]

    # Fill in the counts
    idx = 0
    for i in range(H):
        row_count[i] = W  # initially, row i has W cookies
        for j in range(W):
            c = cookies[i][j]
            c_idx = ord(c) - ord('a')
            row_color_counts[i][c_idx] += 1
            col_color_counts[j][c_idx] += 1

    for j in range(W):
        col_count[j] = H  # initially, column j has H cookies

    # We'll use a queue of (type, index),
    # type=0 => row, type=1 => column
    queue = deque()
    # Initially push all rows and columns
    for i in range(H):
        queue.append((0, i))
    for j in range(W):
        queue.append((1, j))

    while queue:
        t, idx_ = queue.popleft()
        if t == 0:
            # row
            i = idx_
            if row_removed[i]:
                continue
            # check if row_count[i] >= 2 and uniform
            if row_count[i] >= 2:
                # see if there's a color c such that row_color_counts[i][c] == row_count[i]
                rc = row_count[i]
                found_color = -1
                for cidx in range(26):
                    if row_color_counts[i][cidx] == rc:
                        found_color = cidx
                        break
                if found_color != -1:
                    # this row is uniform in color 'found_color', remove row i
                    row_removed[i] = True
                    # remove each cookie in row i from its column
                    # row_count[i] goes to 0, but we'll just mark it removed, or set it explicitly
                    row_count[i] = 0
                    # decrement col_count and col_color_counts for each column j
                    # in range(W) if that column is not removed
                    row_color_counts[i] = [0]*26  # no cookies left
                    for j in range(W):
                        if not col_removed[j]:
                            cidx = ord(cookies[i][j]) - ord('a')
                            col_color_counts[j][cidx] -= 1
                            col_count[j] -= 1
                            # If col_count[j] >=2, re-check that column
                            if col_count[j] >= 2:
                                queue.append((1, j))

        else:
            # column
            j = idx_
            if col_removed[j]:
                continue
            # check if col_count[j] >=2 and uniform
            if col_count[j] >= 2:
                cc = col_count[j]
                found_color = -1
                for cidx in range(26):
                    if col_color_counts[j][cidx] == cc:
                        found_color = cidx
                        break
                if found_color != -1:
                    # this column is uniform, remove col j
                    col_removed[j] = True
                    col_count[j] = 0
                    col_color_counts[j] = [0]*26
                    # remove each cookie in column j from its row
                    for i in range(H):
                        if not row_removed[i]:
                            cidx = ord(cookies[i][j]) - ord('a')
                            row_color_counts[i][cidx] -= 1
                            row_count[i] -= 1
                            if row_count[i] >= 2:
                                queue.append((0, i))

    # Finally, count how many cookies remain
    # This is the sum of row_count[i] for i in [0..H-1]
    answer = sum(row_count)
    print(answer)

# Call main() to comply with the requirement.
main()