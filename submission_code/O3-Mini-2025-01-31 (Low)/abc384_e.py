def main():
    import sys, heapq
    input = sys.stdin.readline

    # Read input
    H, W, X = map(int, input().split())
    P, Q = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(H)]

    # Using zero-indexing for easier handling.
    P -= 1
    Q -= 1

    # visited[i][j] indicates whether cell (i, j) is already part of the group of slimes 
    # that have been absorbed (or that Takahashi has moved onto).
    visited = [[False] * W for _ in range(H)]
    
    # Takahashi starts at (P, Q) with strength equal to the slime in that cell.
    current_strength = grid[P][Q]
    visited[P][Q] = True

    # We maintain a min-heap of boundary candidates that are adjacent to any cell in our group.
    # Each candidate is stored as (slime_strength, i, j).
    heap = []
    
    # Directions for adjacent cells: up, down, left, right.
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    # Function to add unvisited neighbors of cell (i,j) into the heap.
    def add_neighbors(i, j):
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
                visited[ni][nj] = True
                heapq.heappush(heap, (grid[ni][nj], ni, nj))
    
    # Start by adding the neighbors of the starting cell.
    add_neighbors(P, Q)
    
    # Note: The absorption condition given is that a candidate slime with strength v
    # can be absorbed if v < (current_strength / X). To avoid floating point handling,
    # we equivalently require: v * X < current_strength.
    while heap:
        v, i, j = heap[0]  # get the candidate with the smallest strength
        if v * X < current_strength:
            # If the candidate qualifies, absorb it.
            heapq.heappop(heap)
            current_strength += v  # Increase Takahashi's strength
            add_neighbors(i, j)    # Add the neighbors of the newly absorbed slime.
        else:
            # Since the heap is a min-heap, if the smallest candidate cannot be absorbed,
            # then none of the candidates in the heap can be absorbed at the current strength.
            break

    sys.stdout.write(str(current_strength))


if __name__ == "__main__":
    main()