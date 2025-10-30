import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    # 1-indexed permutation
    P = [0] + [int(next(it)) for _ in range(N)]

    visited = [False] * (N + 1)
    result = [0] * (N + 1)

    # cache for pow(2, K, L) for identical cycle lengths
    pow_cache = {}

    for start in range(1, N + 1):
        if visited[start]:
            continue
        # collect one cycle
        cycle = []
        cur = start
        while not visited[cur]:
            visited[cur] = True
            cycle.append(cur)
            cur = P[cur]

        L = len(cycle)
        if L == 0:
            continue

        # compute 2^K mod L, reuse if length repeats
        if L not in pow_cache:
            pow_cache[L] = pow(2, K, L)
        shift = pow_cache[L]

        # apply the shift produced by P^(2^K)
        for idx, node in enumerate(cycle):
            result[node] = cycle[(idx + shift) % L]

    # output
    sys.stdout.write(' '.join(str(result[i]) for i in range(1, N + 1)) + '
')

if __name__ == "__main__":
    sys.setrecursionlimit(1 << 25)
    main()