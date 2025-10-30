import sys

# ---------- helpers ----------
def dist_mod(a: int, b: int, m: int) -> int:
    """minimum number of ±1 (mod m) steps to turn a into b"""
    d = (b - a) % m
    return d if d <= m - d else m - d


def main() -> None:
    it = map(int, sys.stdin.read().strip().split())
    try:
        N = next(it)
    except StopIteration:          # empty input
        return
    M = next(it)

    A = [next(it) for _ in range(N)]
    B = [next(it) for _ in range(N)]

    # ---- special case  M = 2 -----------------------------------------------
    if M == 2:
        # with only two colours no element can be changed without
        # violating the “good” condition, so A must already equal B.
        print(0 if A == B else -1)
        return

    # ----  general case M >= 3 ----------------------------------------------
    # 1.  basic cost : each element travels along the shorter arc
    basic = 0
    for a, b in zip(A, B):
        basic += dist_mod(a, b, M)

    # 2.  “swap” edges create 2-cycles of dependencies.
    #     Edge i (between positions i and i+1, 0-based) is a swap edge iff
    #        A[i]  == B[i+1]  and  A[i+1] == B[i]   (and both vertices change).
    swap = [False] * (N - 1)
    for i in range(N - 1):
        if A[i] != B[i] and A[i + 1] != B[i + 1] \
           and A[i] == B[i + 1] and A[i + 1] == B[i]:
            swap[i] = True

    # minimum number of vertices that have to do an extra “detour” step
    # equals a minimum vertex cover of the swap-edge sub-graph.
    # On a path this is obtained greedily: always take the right end
    # of the leftmost uncovered swap edge.
    extra = 0
    i = 0
    while i < N - 1:
        if swap[i]:
            extra += 1   # choose vertex i+1
            i += 2       # edge i (and possibly i+1) is now covered
        else:
            i += 1

    print(basic + extra)


if __name__ == "__main__":
    main()