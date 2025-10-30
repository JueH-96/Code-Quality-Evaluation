import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    H, W, Q = map(int, sys.stdin.readline().split())
    N = H * W                              # number of cells (1-indexed)
    
    # Four DSU parent arrays.
    #   parL : first (alive) wall on or to the left
    #   parR : first (alive) wall on or to the right
    #   parU : first (alive) wall on or above
    #   parD : first (alive) wall on or below
    #
    # Index 0 acts as a sentinel meaning “no wall exists in that direction”.
    parL = list(range(N + 1))
    parR = list(range(N + 1))
    parU = list(range(N + 1))
    parD = list(range(N + 1))

    removed = [False] * (N + 1)            # True  ⇔  this wall has been destroyed
    destroyed = 0                          # number of destroyed walls

    def find(par, x):
        """DSU find with path compression (sentinel 0 is its own root)."""
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def remove(cell_id: int):
        """Destroy the wall at `cell_id` if it is still present."""
        nonlocal destroyed
        if cell_id == 0 or removed[cell_id]:
            return
        removed[cell_id] = True
        destroyed += 1

        row = (cell_id - 1) // W + 1       # 1-indexed row
        col = (cell_id - 1) %  W + 1       # 1-indexed column

        # update DSU parents so future finds skip over this removed wall
        # left
        if col == 1:
            parL[cell_id] = 0
        else:
            parL[cell_id] = find(parL, cell_id - 1)

        # right
        if col == W:
            parR[cell_id] = 0
        else:
            parR[cell_id] = find(parR, cell_id + 1)

        # up
        if row == 1:
            parU[cell_id] = 0
        else:
            parU[cell_id] = find(parU, cell_id - W)

        # down
        if row == H:
            parD[cell_id] = 0
        else:
            parD[cell_id] = find(parD, cell_id + W)

    # process queries
    for _ in range(Q):
        r, c = map(int, sys.stdin.readline().split())
        idx = (r - 1) * W + c

        if not removed[idx]:
            # wall still exists at bomb position
            remove(idx)
        else:
            # the four independent directional destructions
            if r > 1:
                up_target = find(parU, idx - W)
                remove(up_target)

            if r < H:
                down_target = find(parD, idx + W)
                remove(down_target)

            if c > 1:
                left_target = find(parL, idx - 1)
                remove(left_target)

            if c < W:
                right_target = find(parR, idx + 1)
                remove(right_target)

    remaining = N - destroyed
    print(remaining)

if __name__ == "__main__":
    main()