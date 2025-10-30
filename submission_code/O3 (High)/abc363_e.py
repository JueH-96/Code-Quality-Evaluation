import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    readline = sys.stdin.readline

    H, W, Y = map(int, readline().split())

    # buckets of cells for every elevation (only until Y)
    buckets = [[] for _ in range(Y + 1)]

    # read the grid, fill the buckets
    for r in range(H):
        row = list(map(int, readline().split()))
        base = r * W
        for c, h in enumerate(row):
            if h <= Y:                       # only cells that may get flooded
                buckets[h].append(base + c)

    N = H * W                              # number of cells

    parent   = [-1] * N                    # DSU parent, -1 = not “wet” yet
    comp_sz  = [0]  * N                    # component size (valid only for roots)
    touches  = [False] * N                 # does the component touch the sea?

    # DSU helpers -------------------------------------------------------------
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            return 0

        # union by size
        if comp_sz[ra] < comp_sz[rb]:
            ra, rb = rb, ra

        delta = 0
        if touches[ra] and not touches[rb]:
            delta = comp_sz[rb]
        elif touches[rb] and not touches[ra]:
            delta = comp_sz[ra]

        parent[rb] = ra
        comp_sz[ra] += comp_sz[rb]
        touches[ra] = touches[ra] or touches[rb]
        return delta
    # -------------------------------------------------------------------------

    total_cells = N
    flooded = 0
    answers = [0] * (Y + 1)

    for level in range(1, Y + 1):
        for idx in buckets[level]:
            # activate the cell
            parent[idx] = idx
            comp_sz[idx] = 1
            r, c = divmod(idx, W)
            boundary = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
            touches[idx] = boundary
            if boundary:
                flooded += 1

            # merge with already–wet neighbours
            if r and parent[idx - W] != -1:          # up
                flooded += union(idx, idx - W)
            if r + 1 < H and parent[idx + W] != -1:  # down
                flooded += union(idx, idx + W)
            if c and parent[idx - 1] != -1:          # left
                flooded += union(idx, idx - 1)
            if c + 1 < W and parent[idx + 1] != -1:  # right
                flooded += union(idx, idx + 1)

        answers[level] = total_cells - flooded

    sys.stdout.write('
'.join(map(str, answers[1:])))


if __name__ == "__main__":
    main()