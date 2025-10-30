def main():
    import sys
    data = sys.stdin.read().strip().split()
    H, W, Q = map(int, data[:3])
    queries = data[3:]

    # Total number of cells
    n = H * W

    # hasWall[x] indicates whether cell with index x still has a wall
    hasWall = [True] * n

    # The following arrays implement "linked list" neighbors for each cell:
    # up[x]:    index of the nearest cell above x in the same column that still has a wall
    # down[x]:  nearest cell below x in the same column
    # left[x]:  nearest cell left of x in the same row
    # right[x]: nearest cell right of x in the same row
    up    = [0] * n
    down  = [0] * n
    left_ = [0] * n
    right_ = [0] * n

    # Build initial links (all cells have walls initially).
    # We'll use 0-based indexing for rows and columns:
    # cell (i, j) will correspond to index x = i * W + j.
    for i in range(H):
        for j in range(W):
            x = i * W + j
            if i == 0:
                up[x] = -1
            else:
                up[x] = x - W  # the cell directly above in the same column

            if i == H - 1:
                down[x] = -1
            else:
                down[x] = x + W  # the cell directly below in the same column

            if j == 0:
                left_[x] = -1
            else:
                left_[x] = x - 1  # the cell directly to the left in the same row

            if j == W - 1:
                right_[x] = -1
            else:
                right_[x] = x + 1  # the cell directly to the right in the same row

    # Keep track of how many walls remain.
    remain = n

    # A helper function to remove a wall at index x (if it still exists),
    # and update the linked-list pointers so that future queries skip this cell.
    def removeWall(x):
        nonlocal remain
        if not hasWall[x]:
            return  # already removed
        hasWall[x] = False
        remain -= 1

        u = up[x]
        d = down[x]
        l = left_[x]
        r = right_[x]

        # Link around this cell so it is skipped in future "up"/"down"/"left"/"right" traversals
        if u != -1:
            down[u] = d
        if d != -1:
            up[d] = u
        if l != -1:
            right_[l] = r
        if r != -1:
            left_[r] = l

    # Process each query
    idx_q = 0
    for _ in range(Q):
        r_q = int(queries[idx_q])
        c_q = int(queries[idx_q + 1])
        idx_q += 2

        # Convert 1-based (r_q, c_q) to 0-based index x
        x = (r_q - 1) * W + (c_q - 1)

        if hasWall[x]:
            # Case 1: If there's a wall at (r_q, c_q), destroy it and stop
            removeWall(x)
        else:
            # Case 2: No wall at (r_q, c_q),
            # destroy the first wall up, down, left, and right (if any).
            u = up[x]
            d = down[x]
            l = left_[x]
            r = right_[x]
            # Put them in a set (to avoid duplicates if they coincide)
            candidates = set()
            if u != -1: candidates.add(u)
            if d != -1: candidates.add(d)
            if l != -1: candidates.add(l)
            if r != -1: candidates.add(r)

            for cell in candidates:
                removeWall(cell)

    # Print the remaining number of walls
    print(remain)

# Do not forget to call main()
if __name__ == "__main__":
    main()