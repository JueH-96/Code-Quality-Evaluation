def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))

    # Convert P to 0-based function f
    f = [p - 1 for p in P]

    visited = [False] * N
    final_arr = [0] * N

    for i in range(N):
        if not visited[i]:
            # find the cycle starting from i
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = f[cur]

            L = len(cycle)
            # Compute how many steps to move in this cycle: 2^K mod L
            offset = pow(2, K, L)

            for idx in range(L):
                final_arr[cycle[idx]] = cycle[(idx + offset) % L]

    # Convert back to 1-based
    for i in range(N):
        final_arr[i] += 1

    print(*final_arr)

# Do not forget to call main()
if __name__ == "__main__":
    main()