import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    K = int(next(it))
    P = [int(next(it)) - 1 for _ in range(n)]
    visited = [False] * n
    ans = [0] * n

    for i in range(n):
        if not visited[i]:
            # collect the current cycle starting from i
            cycle = []
            v = i
            while not visited[v]:
                visited[v] = True
                cycle.append(v)
                v = P[v]
            L = len(cycle)
            # effective exponent = 2^K mod L
            e = pow(2, K, L)
            # apply f^e on the cycle
            for j, u in enumerate(cycle):
                ans[u] = cycle[(j + e) % L]

    # output the resulting permutation (1-indexed)
    out = ' '.join(str(x + 1) for x in ans)
    sys.stdout.write(out)


if __name__ == "__main__":
    main()