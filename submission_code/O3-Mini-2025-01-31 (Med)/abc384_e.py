def main():
    import sys
    import heapq

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    X = int(next(it))
    # Convert starting indices to 0-indexed.
    P = int(next(it)) - 1
    Q = int(next(it)) - 1

    # Read the grid.
    grid = []
    for _ in range(H):
        row = [int(next(it)) for _ in range(W)]
        grid.append(row)

    # visited[i][j] will mark that the slime at (i,j) is already absorbed (thus part of Takahashi’s region)
    visited = [[False] * W for _ in range(H)]
    visited[P][Q] = True
    current_strength = grid[P][Q]

    # We use a min-heap (priority queue) to manage "frontier" cells (neighbors of the region).
    # Although our absorption condition grows more lenient as current_strength increases,
    # using a min-heap keyed by cell strength guarantees that if the smallest candidate is
    # not absorbable, none of the larger ones are absorbable at that moment.
    heap = []
    # Use an "in_heap" matrix to avoid pushing the same cell multiple times.
    in_heap = [[False] * W for _ in range(H)]

    # A helper to add a neighbor cell (i,j) to the heap, if valid.
    def add_frontier(i, j):
        if 0 <= i < H and 0 <= j < W and not visited[i][j] and not in_heap[i][j]:
            in_heap[i][j] = True
            heapq.heappush(heap, (grid[i][j], i, j))

    # Directions: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Initialize the frontier with neighbors of the starting cell.
    for di, dj in directions:
        ni, nj = P + di, Q + dj
        add_frontier(ni, nj)

    # The absorption rule is: among adjacent slimes, Takahashi can absorb a slime
    # if its strength s is strictly less than (1/X) * (current_strength),
    # equivalently if current_strength > X * s.
    # Note: When Takahashi absorbs a slime, he moves to that cell and the region (and its frontier)
    # expands by including the new neighbors.
    while heap:
        # Since the heap is sorted by cell strength (the smallest strength first),
        # let (s, i, j) be the candidate with minimum strength.
        s, i, j = heap[0]
        if current_strength > X * s:
            # The candidate qualifies; pop it and absorb.
            heapq.heappop(heap)
            # Check if the cell was already absorbed (it might have been added more than once).
            if visited[i][j]:
                continue
            visited[i][j] = True
            current_strength += s
            # Add its neighbors to the frontier.
            for di, dj in directions:
                ni, nj = i + di, j + dj
                add_frontier(ni, nj)
        else:
            # Since the smallest candidate in the frontier does not satisfy the absorption condition,
            # none of the remaining candidates (which all have strength >= s) can be absorbed.
            break

    sys.stdout.write(str(current_strength))

if __name__ == '__main__':
    main()