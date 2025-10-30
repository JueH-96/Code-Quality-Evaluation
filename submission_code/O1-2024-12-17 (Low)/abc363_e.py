def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    H, W, Y = map(int, input_data[:3])
    A = list(map(int, input_data[3:]))

    # We will solve this by sorting all cells in ascending order of elevation
    # and using a Union-Find (Disjoint Set Union, DSU) structure.
    # As we increase sea level L from 1..Y, we "activate" all cells whose elevation <= L.
    # Then we union these cells with their neighbors that are already activated.
    # If the connected component of any activated cell touches a boundary,
    # that entire component is considered "sunken" (sea).
    #
    # Key idea: a cell sinks if it is connected (through cells of elevation <= L) to the boundary.
    # We'll keep track of the size of each connected component. If a component is marked "sea,"
    # all its cells are sunk. The number of cells above water = total_cells - number_of_sunk_cells.
    #
    # We'll process the cells in ascending elevation order, keep a pointer into that sorted list.
    # For each L = 1..Y, we union in all cells whose elevation <= L, and if the newly formed
    # component touches any boundary (or merges with a boundary-sunk component), it becomes sea.
    # We accumulate results for each L.
    #
    # Implementation details:
    # 1) Flatten cells into (elev, r, c) and sort by elev.
    # 2) Union-Find with an array to mark whether a root is "sea" and the size of each component.
    # 3) As we activate cells, we union them with already activated neighbors.
    #    If the unioned component touches boundary, mark it as sea. Then we update
    #    the global count of sunk cells accordingly.
    # 4) For each L in 1..Y, store total_cells - sunk_cells in the output.
    #
    # Complexity: O(H*W log(H*W) + (H*W + Y) α(H*W)) which is feasible in optimized Python
    # for H,W <= 1000 (up to 10^6 cells). Carefully written code should pass.

    # Quick adjacency helper
    def neighbors(r, c):
        # yields valid 4-directional neighbors
        if r > 0:
            yield r-1, c
        if r < H-1:
            yield r+1, c
        if c > 0:
            yield r, c-1
        if c < W-1:
            yield r, c+1

    # DSU (Union-Find) implementation
    parent = list(range(H*W))
    size = [1]*(H*W)    # size of the connected component (for the root)
    is_sea = [False]*(H*W)  # mark if the component is sea (touches boundary)
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        # Union by size
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        # now ra is the bigger root
        parent[rb] = ra
        # if the smaller root was sea or bigger root was sea, or any boundary indicator -> sea
        was_sea_before = is_sea[ra]
        new_sea_state = is_sea[ra] or is_sea[rb]
        size[ra] += size[rb]
        is_sea[ra] = new_sea_state
        return ra

    def idx(r, c):
        return r*W + c

    total_cells = H*W
    # We'll sort the cells by elevation
    cells = []
    p = 0  # pointer in the sorted cells
    for r in range(H):
        for c in range(W):
            cells.append((A[r*W + c], r, c))
    cells.sort(key=lambda x: x[0])

    # processed[] indicates if a cell is active (i.e. elevation <= current L).
    processed = [False]*(H*W)

    # We'll iterate L=1..Y. For each L, we activate all new cells with elev <= L.
    # Then union them with processed neighbors. If any cell is boundary, mark that comp sea.
    # Keep a global count of how many cells are sunk: sum_sea.
    sum_sea = 0

    results = [0]*Y

    # A helper to mark a root as sea (and update sum_sea if that was newly turned to sea)
    def mark_sea(root):
        nonlocal sum_sea
        if not is_sea[root]:
            is_sea[root] = True
            sum_sea += size[root]

    # We also need a quick check for boundary
    def is_boundary(r, c):
        return (r == 0 or r == H-1 or c == 0 or c == W-1)

    n = len(cells)

    # We'll now process sea levels from 1..Y
    elev_ptr = 0  # pointer over sorted cells
    for level in range(1, Y+1):
        # Activate all cells whose elevation <= level
        while elev_ptr < n and cells[elev_ptr][0] <= level:
            e, r, c = cells[elev_ptr]
            elev_ptr += 1
            cell_index = idx(r, c)
            processed[cell_index] = True

            # Initially, it is in its own set, so check if boundary -> mark sea
            root_now = find(cell_index)
            if is_boundary(r, c):
                mark_sea(root_now)

            # Union with processed neighbors
            for nr, nc in neighbors(r, c):
                neighbor_index = idx(nr, nc)
                if processed[neighbor_index]:
                    root_before = find(neighbor_index)
                    root_now = find(cell_index)
                    # If both are already in the same component, do nothing
                    if root_now != root_before:
                        sea_before_size = 0
                        # if either is sea, we might reduce sum_sea then re-add
                        # to avoid double counting. We'll handle it systematically:
                        old_root_now = root_now
                        old_root_before = root_before
                        if is_sea[old_root_now]:
                            sea_before_size += size[old_root_now]
                        if is_sea[old_root_before] and old_root_before != old_root_now:
                            sea_before_size += size[old_root_before]
                        new_root = union(root_now, root_before)
                        # Now new_root is the root after union
                        if is_sea[new_root]:
                            # Recalculate how many squares are in the new_root
                            # Then we add to sum_sea the difference
                            sea_after_size = size[new_root]
                            # The difference is sea_after_size - sea_before_size
                            sum_sea_diff = sea_after_size - sea_before_size
                            if sum_sea_diff > 0:
                                sum_sea += sum_sea_diff
                        else:
                            # If the new root is boundary => mark sea
                            # but let's do it carefully
                            pass

                        # Possibly if new_root is boundary from either part
                        if is_sea[new_root] == False:
                            # We must check if boundary is triggered
                            # Because if any cell in the component is boundary, we do is_sea[new_root] = True
                            # We'll just rely on boundary check from each newly activated cell
                            # Actually, it's simpler to do an explicit boundary check for neighbors:
                            br1 = is_boundary(r, c)
                            br2 = is_boundary(nr, nc)
                            # If the neighbor was already known boundary, is_sea[find(neighbor_index)] might be True
                            # but that would have merged. We'll explicitly check:
                            if br1 or br2 or is_sea[find(neighbor_index)] or is_sea[new_root]:
                                # Mark the new root as sea
                                mark_sea(find(cell_index))
            # end for each neighbor
        # end while - activation

        # Now the number of squares that remain above water is total_cells - sum_sea
        results[level-1] = total_cells - sum_sea

    # Output results
    print('
'.join(map(str, results)))


# Don't forget to call main()
if __name__ == "__main__":
    main()