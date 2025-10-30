def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N, K = map(int, input().split())
    P = list(map(int, input().split()))
    # P is 1-based mapping: i -> P[i-1]

    ans = [0] * N
    visited = [False] * N

    for i in range(N):
        if not visited[i]:
            # extract the cycle starting at i
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = P[cur] - 1
            L = len(cycle)
            # compute how far to rotate: 2^K mod L
            # moving each element forward by r steps in the cycle
            r = pow(2, K, L)
            # assign answers
            for j, idx in enumerate(cycle):
                ans[idx] = cycle[(j + r) % L] + 1

    # output
    print(" ".join(map(str, ans)))

if __name__ == "__main__":
    main()