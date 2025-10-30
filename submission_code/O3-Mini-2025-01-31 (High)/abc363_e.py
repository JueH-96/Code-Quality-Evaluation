def main():
    import sys,sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    try:
        H = int(next(it))
        W = int(next(it))
        Y_query = int(next(it))
    except StopIteration:
        return
    n = H * W
    # Read grid cells in row–major order.
    grid = [0] * n
    for i in range(n):
        grid[i] = int(next(it))
    # Build a list of cells: each item is (elevation, row, col).
    cells = [None] * n
    idx = 0
    for i in range(H):
        base = i * W
        for j in range(W):
            cells[idx] = (grid[base+j], i, j)
            idx += 1
    cells.sort(key=lambda x: x[0])
    cells_len = len(cells)
    
    # Union–find arrays. For each cell (flattened index) we will track:
    #     parent, size and a flag is_border (True if that active cell/component touches the island boundary).
    parent = [0] * n
    size = [0] * n
    is_border = [False] * n
    active = [False] * n  # indicates if the cell has been "added" (activated).
    for i in range(n):
        parent[i] = i
        size[i] = 0   # set to 1 upon activation.
        is_border[i] = False
        active[i] = False

    # Helper functions for union–find.
    def find(x):
        # Path halving.
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return 0
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        # Calculate 'before' flooded count contributions.
        before = (size[rx] if is_border[rx] else 0) + (size[ry] if is_border[ry] else 0)
        parent[ry] = rx
        size[rx] += size[ry]
        is_border[rx] = is_border[rx] or is_border[ry]
        after = size[rx] if is_border[rx] else 0
        return after - before

    # We now simulate the sea level rising.
    # global_flooded will count the number of activated cells that belong to
    # a union–find component which touches the border (i.e. flooded).
    global_flooded = 0
    total = n
    results = [None] * Y_query
    p = 0  # pointer into sorted cells
    # Process sea level (years) from 1 to Y_query.
    for sea in range(1, Y_query + 1):
        # Activate all cells having elevation <= current sea level.
        while p < cells_len and cells[p][0] <= sea:
            elev, i, j = cells[p]
            p += 1
            pos = i * W + j
            active[pos] = True
            parent[pos] = pos
            size[pos] = 1
            # Mark border – note: a cell touches the ocean if it lies on the edge.
            if i == 0 or i == H - 1 or j == 0 or j == W - 1:
                is_border[pos] = True
            else:
                is_border[pos] = False
            if is_border[pos]:
                global_flooded += 1
            # Check the 4 neighbors; if they are activated, union them.
            # Up
            if i > 0:
                npos = (i - 1) * W + j
                if active[npos]:
                    delta = union(pos, npos)
                    global_flooded += delta
            # Down
            if i < H - 1:
                npos = (i + 1) * W + j
                if active[npos]:
                    delta = union(pos, npos)
                    global_flooded += delta
            # Left
            if j > 0:
                npos = i * W + (j - 1)
                if active[npos]:
                    delta = union(pos, npos)
                    global_flooded += delta
            # Right
            if j < W - 1:
                npos = i * W + (j + 1)
                if active[npos]:
                    delta = union(pos, npos)
                    global_flooded += delta
        # At this sea level, the cells that are flooded are all activated cells
        # which are in a border–connected component. The remaining area is:
        remain = total - global_flooded
        results[sea - 1] = str(remain)
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()