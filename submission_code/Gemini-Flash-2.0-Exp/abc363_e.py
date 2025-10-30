def solve():
    H, W, Y = map(int, input().split())
    A = []
    for _ in range(H):
        A.append(list(map(int, input().split())))

    for year in range(1, Y + 1):
        sunk = set()
        
        # Find initial sunk cells
        for r in range(H):
            for c in range(W):
                if A[r][c] <= year:
                    if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                        sunk.add((r, c))

        newly_sunk = sunk.copy()
        
        while newly_sunk:
            next_newly_sunk = set()
            for r, c in newly_sunk:
                # Check neighbors
                neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
                for nr, nc in neighbors:
                    if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in sunk and A[nr][nc] <= year:
                        sunk.add((nr, nc))
                        next_newly_sunk.add((nr, nc))
            newly_sunk = next_newly_sunk
        
        print(H * W - len(sunk))

solve()