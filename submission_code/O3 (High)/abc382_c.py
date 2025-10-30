import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    it = iter(data)
    N = next(it)
    M = next(it)

    A = [next(it) for _ in range(N)]
    B = [next(it) for _ in range(M)]

    # Largest value we need to care about
    V = max(max(A), max(B))

    INF = N + 5                              # larger than any possible index
    earliest = [INF] * (V + 1)               # earliest[x] = first person whose A_i == x

    # record earliest appearance of each gourmet level
    for idx, val in enumerate(A, 1):         # people are 1-indexed
        if earliest[val] > idx:
            earliest[val] = idx

    # build prefix minima so that earliest[x] = min{i | A_i <= x}
    for x in range(1, V + 1):
        if earliest[x] > earliest[x - 1]:
            earliest[x] = earliest[x - 1]

    out_lines = []
    for val in B:
        idx = earliest[val]
        out_lines.append(str(idx) if idx <= N else "-1")

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()