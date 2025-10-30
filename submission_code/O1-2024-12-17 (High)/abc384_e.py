def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    H, W, X = map(int, data[:3])
    P, Q = map(int, data[3:5])
    S_flat = list(map(int, data[5:]))

    # Convert to 0-based indices
    P -= 1
    Q -= 1

    # Build the 2D strength array
    S = [S_flat[i*W:(i+1)*W] for i in range(H)]

    # Boolean arrays to mark which cells are already in Takahashi's "region"
    # and which cells are already in the adjacency list (min-heap).
    in_region = [[False]*W for _ in range(H)]
    in_adj = [[False]*W for _ in range(H)]

    # Takahashi's initial strength:
    T = S[P][Q]

    # Mark the starting cell as absorbed (in-region).
    in_region[P][Q] = True

    import heapq
    adj = []  # min-heap of (slime_strength, row, col)

    # A helper to add a neighbor to the adjacency heap if not already in region/adj
    def try_add(r, c):
        if 0 <= r < H and 0 <= c < W:
            if not in_region[r][c] and not in_adj[r][c]:
                in_adj[r][c] = True
                heapq.heappush(adj, (S[r][c], r, c))

    # Initially add neighbors of (P, Q) to the heap
    try_add(P-1, Q)
    try_add(P+1, Q)
    try_add(P, Q-1)
    try_add(P, Q+1)

    # While there are adjacent slimes we can consider absorbing:
    while adj:
        strength, r, c = heapq.heappop(adj)
        # If this cell was absorbed in the meantime, skip it
        if in_region[r][c]:
            continue

        # Check if we can absorb this slime
        # Condition: strength * X < T   (strictly less than T/X means T > X*strength)
        if strength * X < T:
            # Absorb it:
            T += strength
            in_region[r][c] = True

            # Now add its neighbors to the heap (they become adjacent to the region)
            try_add(r-1, c)
            try_add(r+1, c)
            try_add(r, c-1)
            try_add(r, c+1)
        else:
            # Since the heap is a min-heap, if we cannot absorb the smallest
            # remaining slime, we cannot absorb any larger ones. Stop.
            break

    print(T)

# Don't forget to call main!
if __name__ == "__main__":
    main()