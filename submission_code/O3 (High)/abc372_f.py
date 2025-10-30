import sys

MOD = 998244353

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))

    # store extra edges in 0-based indexing
    extra = [(int(next(it)) - 1, int(next(it)) - 1) for _ in range(M)]

    # trivial case : no extra edges  -> only one possible path
    if M == 0:
        print(1)
        return

    counts = [0] * N           # counts array (rotating)
    counts[0] = 1              # start at vertex 0 (which is 1 in 1-based)
    offset = 0                 # current rotation offset

    for _ in range(K):
        contrib = []           # (target_vertex, value) pairs

        # gather contributions produced by extra edges
        for x, y in extra:
            idx_src = (x - offset) % N
            val = counts[idx_src]
            if val:
                contrib.append((y, val))

        # apply ring shift (all paths follow the ring edge)
        offset = (offset + 1) % N

        # add contributions of extra edges to counts after shift
        for y, val in contrib:
            idx_tgt = (y - offset) % N
            counts[idx_tgt] = (counts[idx_tgt] + val) % MOD

    # answer is total number of paths (sum over all vertices)
    print(sum(counts) % MOD)


if __name__ == "__main__":
    main()