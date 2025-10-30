import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        adj[u].append(v)
        adj[v].append(u)
    W = [int(next(it)) for _ in range(N)]
    A = [int(next(it)) for _ in range(N)]
    # Prepare nodes in increasing W order
    idxs = list(range(N))
    idxs.sort(key=lambda x: W[x])
    # DP array f[x] = max operations starting from one piece at x
    f = [0] * N
    # We'll reuse a dp array up to size max(W)-1
    maxW = max(W)
    # Pre-allocate a single dp and reuse slice
    dp = [0] * maxW  # dp[c] for current capacity
    for x in idxs:
        cap = W[x] - 1
        if cap < 0:
            # cannot even remove current piece (but W[x]>=1 by constraints)
            f[x] = 1
            continue
        # reset dp[0..cap]
        for i in range(cap + 1):
            dp[i] = 0
        # 0-1 knapsack over neighbors y with W[y] < W[x]
        # item weight = W[y], value = f[y]
        for y in adj[x]:
            wy = W[y]
            if wy < W[x]:
                vy = f[y]
                # roll backwards
                # for c from cap down to wy: dp[c] = max(dp[c], dp[c-wy] + vy)
                # local vars for speed
                _dp = dp
                for c in range(cap, wy - 1, -1):
                    val = _dp[c - wy] + vy
                    if val > _dp[c]:
                        _dp[c] = val
        # best we can do is 1 (for removing this piece) + best dp
        best = 0
        # find max dp value
        for v in dp[:cap+1]:
            if v > best:
                best = v
        f[x] = 1 + best
    # Now sum over initial pieces
    ans = 0
    for i in range(N):
        if A[i]:
            ans += A[i] * f[i]
    print(ans)

if __name__ == "__main__":
    main()