def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    H, W, Y = map(int, input_data[:3])
    A_vals = list(map(int, input_data[3:]))

    # Reshape A into 2D
    A = [A_vals[i*W : (i+1)*W] for i in range(H)]

    # We'll use a modified D'Johnson/Prim-like algorithm from the "sea boundary"
    # to compute the minimal sea level ("flood level") at which each cell gets flooded.
    #
    # Interpretation:
    #  flood_level[r][c] = the smallest L such that there's a path (of cells with
    #  elevation <= L) from (r,c) to the outside boundary.  At sea level L' >= L,
    #  the cell (r,c) is flooded.
    #
    # Steps:
    #  1. Initialize flood_level to "infinity" for all cells.
    #  2. For all boundary cells, flood_level = elevation(A[r][c]), and push them
    #     into a min-heap with that priority.
    #  3. Pop from the heap the cell with the smallest "cost".  If that cost
    #     equals flood_level for that cell, try to update neighbors:
    #       new_cost = max(current_cell_cost, neighbor_elevation).
    #     If new_cost < flood_level[neighbor], update and push into the heap.
    #  4. At the end, flood_level[r][c] is the minimal sea level to drown (r,c).
    #
    #  Then we build a frequency array freq[x] = number of cells with flood_level == x.
    #  A prefix sum of freq says how many cells are drowned at or below sea level L.
    #  For year i, sea level = i; drowned(i) = prefix[i],
    #  area above water = total_cells - drowned(i).

    # Prepare for the priority-queue approach
    INF = 10**10
    flood_level = [ [INF]*W for _ in range(H) ]
    pq = []  # (cost, r, c)

    # Push boundary cells
    def push_boundary(r, c):
        if flood_level[r][c] > A[r][c]:
            flood_level[r][c] = A[r][c]
            heapq.heappush(pq, (A[r][c], r, c))

    # Top/bottom row
    for j in range(W):
        push_boundary(0, j)
        push_boundary(H-1, j)

    # Left/right column
    for i in range(H):
        push_boundary(i, 0)
        push_boundary(i, W-1)

    # Directions for neighbors
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    # D'Johnson/Prim-like spreading
    while pq:
        cost, r, c = heapq.heappop(pq)
        if cost > flood_level[r][c]:
            continue
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                new_cost = max(cost, A[nr][nc])
                if flood_level[nr][nc] > new_cost:
                    flood_level[nr][nc] = new_cost
                    heapq.heappush(pq, (new_cost, nr, nc))

    # Now, flood_level[r][c] tells us the minimal sea level at which (r,c) is drowned.
    # Build a frequency array freq[x] that counts how many cells have flood_level == x.
    maxA = max(A_vals)  # highest possible flood level we need to consider
    freq = [0]*(maxA+1)
    for i in range(H):
        for j in range(W):
            # flood_level should never exceed maxA, but just in case:
            lv = flood_level[i][j]
            if lv > maxA:
                lv = maxA
            freq[lv] += 1

    # Convert freq to prefix so that prefix[x] = number of cells with flood_level <= x
    for x in range(1, maxA+1):
        freq[x] += freq[x-1]

    total_cells = H*W

    # For each year i from 1..Y, sea level = i
    # Drowned cells = freq[min(i, maxA)], area above = total_cells - drowned
    import sys
    o = []
    for i in range(1, Y+1):
        if i <= maxA:
            drowned = freq[i]
        else:
            drowned = freq[maxA]
        o.append(str(total_cells - drowned))

    print("
".join(o))

# Do not forget to call main()
if __name__ == "__main__":
    main()