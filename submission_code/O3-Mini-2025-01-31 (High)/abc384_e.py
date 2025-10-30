def main():
    import sys, heapq
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Parse input. All values are given as bytes so we convert them to int.
    it = map(int, data)
    H = int(next(it))
    W = int(next(it))
    X = int(next(it))
    P = int(next(it))
    Q = int(next(it))
    grid = []
    for _ in range(H):
        row = [int(next(it)) for _ in range(W)]
        grid.append(row)
    
    # Using 0-based indices for rows and columns.
    start_r, start_c = P - 1, Q - 1
    # Takahashi's initial strength
    current_strength = grid[start_r][start_c]
    
    # We maintain a "visited" grid to keep track of cells already absorbed.
    visited = [[False] * W for _ in range(H)]
    # "in_heap" prevents us from pushing the same candidate twice.
    in_heap = [[False] * W for _ in range(H)]
    
    visited[start_r][start_c] = True
    
    # Priority queue for candidate border cells.
    # Each candidate is represented as a tuple: (s, r, c)
    # We always want to consider the candidate with the smallest slime strength s.
    pq = []
    
    # The process: Our region starts as the starting cell.
    # Every move, we can absorb a cell adjacent to our region
    # provided its strength s satisfies s < (current_strength / X), 
    # equivalently: X * s < current_strength.
    # When we absorb it our region expands and our total strength increases.
    
    # For a grid cell (r, c), its neighbors in the grid:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def add_candidate(r, c):
        if 0 <= r < H and 0 <= c < W and (not visited[r][c]) and (not in_heap[r][c]):
            in_heap[r][c] = True
            heapq.heappush(pq, (grid[r][c], r, c))
    
    # Initially add the neighbors of the starting region cell.
    for dr, dc in directions:
        nr, nc = start_r + dr, start_c + dc
        add_candidate(nr, nc)
    
    # Process candidates in order of increasing slime strength.
    while pq:
        s, r, c = heapq.heappop(pq)
        # If this cell is already absorbed, skip it.
        if visited[r][c]:
            continue
        # Check absorption condition:
        # We can absorb a candidate if its strength s is strictly less than (current_strength / X).
        # Equivalently: X * s < current_strength.
        if X * s < current_strength:
            # Absorb the slime: add its strength to current_strength and merge it into our region.
            current_strength += s
            visited[r][c] = True
            # By absorbing, we extend our border: add all new neighbors of (r, c)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                add_candidate(nr, nc)
        else:
            # Because the priority queue gives the smallest candidate,
            # if this candidate is not absorbable, none of the remaining ones can be.
            break

    sys.stdout.write(str(current_strength))

if __name__ == '__main__':
    main()