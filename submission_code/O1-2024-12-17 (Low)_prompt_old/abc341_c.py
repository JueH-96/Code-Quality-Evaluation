def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, N = map(int, input_data[:3])
    T = input_data[3]
    S = input_data[4:]

    # Read the grid and mark land as True, sea as False
    land = [[(ch == '.') for ch in row] for row in S]

    # feasible[r][c] will indicate whether (r,c) is a possible position
    # Initialize with all land cells.
    feasible = [[land[r][c] for c in range(W)] for r in range(H)]

    for move in T:
        # Prepare the next feasible state, start all as False
        next_feasible = [[False]*W for _ in range(H)]

        if move == 'L':
            # (r, c) <- (r, c+1)
            for r in range(H):
                for c in range(W-1):
                    if feasible[r][c+1]:
                        next_feasible[r][c] = True

        elif move == 'R':
            # (r, c) <- (r, c-1)
            for r in range(H):
                for c in range(1, W):
                    if feasible[r][c-1]:
                        next_feasible[r][c] = True

        elif move == 'U':
            # (r, c) <- (r+1, c)
            for r in range(H-1):
                for c in range(W):
                    if feasible[r+1][c]:
                        next_feasible[r][c] = True

        else:  # move == 'D'
            # (r, c) <- (r-1, c)
            for r in range(1, H):
                for c in range(W):
                    if feasible[r-1][c]:
                        next_feasible[r][c] = True

        # Mask with land to ensure we do not step onto sea
        for r in range(H):
            for c in range(W):
                next_feasible[r][c] &= land[r][c]

        feasible = next_feasible

    # Count how many cells remain feasible
    ans = sum(sum(row) for row in feasible)
    print(ans)

def _test():
    # You can put sample tests here if you like, but it won't be called.
    pass

if __name__ == "__main__":
    solve()