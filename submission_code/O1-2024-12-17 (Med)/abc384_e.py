def main():
    import sys
    import heapq

    data = sys.stdin.read().strip().split()
    H, W, X = map(int, data[:3])
    P, Q = map(int, data[3:5])
    S_flat = list(map(int, data[5:]))

    # Convert P, Q to zero-based indices
    P -= 1
    Q -= 1

    # Build the 2D strength array
    S = []
    idx = 0
    for _ in range(H):
        row = S_flat[idx:idx+W]
        idx += W
        S.append(row)

    # Takahashi's initial strength
    T = S[P][Q]

    # Keep track of which cells have been "absorbed"/visited
    visited = [[False]*W for _ in range(H)]
    visited[P][Q] = True

    # Min-heap for frontier slimes (strength, row, col)
    pq = []

    def push_if_valid(r, c):
        if 0 <= r < H and 0 <= c < W and not visited[r][c]:
            heapq.heappush(pq, (S[r][c], r, c))

    # Push neighbors of the starting cell
    if P - 1 >= 0: push_if_valid(P - 1, Q)
    if P + 1 < H:  push_if_valid(P + 1, Q)
    if Q - 1 >= 0: push_if_valid(P, Q - 1)
    if Q + 1 < W:  push_if_valid(P, Q + 1)

    # Absorb process
    while pq:
        strength, r, c = heapq.heappop(pq)
        if visited[r][c]:
            # Already absorbed in a previous step
            continue
        # Check if Takahashi can absorb this slime
        if strength * X < T:
            # Absorb
            T += strength
            visited[r][c] = True
            # Add its neighbors to the frontier
            if r - 1 >= 0: push_if_valid(r - 1, c)
            if r + 1 < H:  push_if_valid(r + 1, c)
            if c - 1 >= 0: push_if_valid(r, c - 1)
            if c + 1 < W:  push_if_valid(r, c + 1)
        else:
            # Cannot absorb this slime, so we cannot absorb any larger ones
            # in the heap either.
            break

    print(T)

# Do not forget to call main()
if __name__ == "__main__":
    main()