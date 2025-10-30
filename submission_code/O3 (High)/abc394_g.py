import sys
sys.setrecursionlimit(1 << 25)

def main() -> None:
    input = sys.stdin.readline

    H, W = map(int, input().split())
    N = H * W                         # number of buildings = number of cells
    F = []                            # height of each building
    for _ in range(H):
        F.extend(map(int, input().split()))

    Q = int(input())
    Y  = [0] * Q                      # start floor
    Z  = [0] * Q                      # goal  floor
    s_idx = [0] * Q                   # start cell index
    t_idx = [0] * Q                   # goal  cell index

    # for every cell store list of query-ids that use the cell as an end-point
    queries_of_cell = [[] for _ in range(N)]

    for qi in range(Q):
        a, b, y, c, d, z = map(int, input().split())
        a -= 1; b -= 1; c -= 1; d -= 1
        idx1 = a * W + b
        idx2 = c * W + d
        s_idx[qi] = idx1
        t_idx[qi] = idx2
        Y[qi]     = y
        Z[qi]     = z
        queries_of_cell[idx1].append(qi)
        queries_of_cell[idx2].append(qi)

    parent = [-1] * N                 # DSU parent (undefined while inactive)
    size   = [1]  * N                 # component size
    active = [False] * N              # has the node already been activated?
    comp_q = [None] * N               # for a root: set of query-ids whose one end is in the component

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # result for each query: G value (largest floor where the two cells are connected)
    Gres = [-1] * Q

    def union(x: int, y: int, h: int) -> None:
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        # merge ry into rx
        parent[ry] = rx
        size[rx] += size[ry]

        set_big = comp_q[rx]
        if set_big is None:
            set_big = set()
            comp_q[rx] = set_big
        set_small = comp_q[ry]
        if set_small:
            for qid in set_small:
                if Gres[qid] != -1:          # already solved
                    continue
                if qid in set_big:
                    Gres[qid] = h            # first moment both ends in same component
                    set_big.discard(qid)
                else:
                    set_big.add(qid)
            comp_q[ry] = None

    # activation of a single cell
    def activate(idx: int, h: int) -> None:
        active[idx] = True
        parent[idx] = idx
        comp_set = set()
        # process queries which use this cell
        for qid in queries_of_cell[idx]:
            if Gres[qid] != -1:
                continue
            if qid in comp_set:
                Gres[qid] = h                # both ends are this very cell
                comp_set.discard(qid)
            else:
                comp_set.add(qid)
        comp_q[idx] = comp_set

        r = idx // W
        c = idx - r * W
        # neighbouring active cells
        if r and active[idx - W]:
            union(idx, idx - W, h)
        if r + 1 < H and active[idx + W]:
            union(idx, idx + W, h)
        if c and active[idx - 1]:
            union(idx, idx - 1, h)
        if c + 1 < W and active[idx + 1]:
            union(idx, idx + 1, h)

    # process all cells descending by height
    order = sorted(range(N), key=lambda i: F[i], reverse=True)
    for idx in order:
        activate(idx, F[idx])

    # every query should be solved now
    # compute minimal number of stair moves
    out_lines = []
    for qi in range(Q):
        y = Y[qi]
        z = Z[qi]
        g = Gres[qi]
        if g >= min(y, z):
            out_lines.append(str(abs(y - z)))
        else:
            out_lines.append(str(y + z - 2 * g))

    print('
'.join(out_lines))

if __name__ == "__main__":
    main()