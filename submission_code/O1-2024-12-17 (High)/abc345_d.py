def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    H = int(input_data[1])
    W = int(input_data[2])
    tiles = []
    idx = 3
    total_area = 0
    for _ in range(N):
        A = int(input_data[idx])
        B = int(input_data[idx+1])
        idx += 2
        tiles.append((A, B))
        total_area += A * B

    # Quick check: if total tile area is less than the grid area => impossible
    if total_area < H * W:
        print("No")
        return

    # -------------------------------------------------------------------------
    # Build all possible placements.
    # A "placement" is (tile_index, list_of_covered_cells)
    # We will store:
    #   placements[p] = list of covered cell-indices (each cell index = r*W + c)
    #   tile_of_placement[p] = i  (which tile it uses)
    #
    # Then for each cell c, we keep a list of placements that cover that cell:
    #   cover_of_cell[c] = list of placement indices p
    # -------------------------------------------------------------------------
    placements = []
    tile_of_placement = []

    for i, (a, b) in enumerate(tiles):
        # Try both orientations (a,b) and (b,a).  Even if a == b, it's okay to list duplicates.
        for (hh, ww) in [(a, b), (b, a)]:
            # If the tile doesn't fit in a given orientation, skip
            if hh > H or ww > W:
                continue
            # Generate all top-left positions where this tile can be placed
            for r in range(H - hh + 1):
                for c in range(W - ww + 1):
                    covered_cells = []
                    for rr in range(r, r + hh):
                        for cc in range(c, c + ww):
                            covered_cells.append(rr * W + cc)
                    placements.append(covered_cells)
                    tile_of_placement.append(i)

    # Now build cover_of_cell: for each cell index, which placements cover it
    cover_of_cell = [[] for _ in range(H * W)]
    for p, covered_cells in enumerate(placements):
        for c in covered_cells:
            cover_of_cell[c].append(p)

    # We will do a backtracking "exact cover" approach:
    #   - covered[c] = whether cell c is currently covered
    #   - used_tile[i] = whether tile i is used
    # We try to cover all H*W cells exactly once.  Tiles can be skipped
    # by never choosing any placement from a certain tile.
    covered = [False] * (H * W)
    used_tile = [False] * N

    def backtrack(covered_count):
        # If all cells are covered, we are done
        if covered_count == H * W:
            return True

        # Choose an uncovered cell.  To speed up, choose one with the fewest
        # feasible placements ("MRV" heuristic).
        best_cell = -1
        best_feasible = None
        best_size = float('inf')

        # Find the next uncovered cell with minimal possible placements
        for cell in range(H * W):
            if not covered[cell]:
                feasible = []
                for p in cover_of_cell[cell]:
                    t = tile_of_placement[p]
                    if not used_tile[t]:
                        # Check if all cells in this placement are free
                        can_place = True
                        for c2 in placements[p]:
                            if covered[c2]:
                                can_place = False
                                break
                        if can_place:
                            feasible.append(p)
                if len(feasible) == 0:
                    # No way to cover this cell
                    return False
                if len(feasible) < best_size:
                    best_size = len(feasible)
                    best_feasible = feasible
                    best_cell = cell
                # If we find a cell with 1 feasible placement, we can't do better in scanning
                if best_size == 1:
                    break

        # Try each feasible placement for the chosen cell
        for p in best_feasible:
            t = tile_of_placement[p]
            # Place this tile
            used_tile[t] = True
            for c2 in placements[p]:
                covered[c2] = True

            if backtrack(covered_count + len(placements[p])):
                return True

            # Revert
            used_tile[t] = False
            for c2 in placements[p]:
                covered[c2] = False

        return False

    # Run backtracking. If success => "Yes", else "No".
    if backtrack(0):
        print("Yes")
    else:
        print("No")