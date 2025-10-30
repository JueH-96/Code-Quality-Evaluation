def solve():
    K, Sx, Sy, Tx, Ty = map(int, input().split())

    def get_tile(x, y):
        ix = x // K
        iy = y // K
        if (ix % 2) == (iy % 2):
            k = int(y - iy * K)
            return ix, iy, k
        else:
            k = int(x - ix * K)
            return ix, iy, k

    Sx_tile, Sy_tile, Sk_start = get_tile(Sx, Sy)
    Tx_tile, Ty_tile, Sk_end = get_tile(Tx, Ty)

    q = [(Sx_tile, Sy_tile, Sk_start, 0)]
    visited = set()
    visited.add((Sx_tile, Sy_tile, Sk_start))

    while q:
        x, y, k, dist = q.pop(0)
        if x == Tx_tile and y == Ty_tile and k == Sk_end:
            print(dist)
            return

        # Possible moves
        moves = []
        if (x % 2) == (y % 2):
            moves.append((x, y, (k + 1) % K))  # Move along k
            moves.append((x + (1 if (x % 2) == 0 else -1), y, k)) #Move along x
            moves.append((x, y + (1 if (y % 2) == 0 else -1), k)) #Move along y

        else:
            moves.append((x, y, (k + 1) % K))  # Move along k
            moves.append((x + (1 if (x % 2) == 0 else -1), y, k)) #Move along x
            moves.append((x, y + (1 if (y % 2) == 0 else -1), k)) #Move along y

        for nx, ny, nk in moves:
            if (nx, ny, nk) not in visited:
                visited.add((nx, ny, nk))
                q.append((nx, ny, nk, dist + 1))


T = int(input())
for _ in range(T):
    solve()