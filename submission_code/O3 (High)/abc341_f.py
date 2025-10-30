import sys


def knap_max_value(capacity: int, items):
    """
    0/1 knapsack : items is list of (weight, value)   (weight <= capacity)
    returns maximal total value achievable with total weight <= capacity
    """
    if capacity == 0 or not items:
        return 0
    dp = [-1] * (capacity + 1)
    dp[0] = 0
    for w, v in items:
        if w > capacity:
            continue
        # traverse backwards to keep 0/1 property
        for c in range(capacity - w, -1, -1):
            if dp[c] >= 0:
                nv = dp[c] + v
                if nv > dp[c + w]:
                    dp[c + w] = nv
    # maximum of reachable states
    return max(dp)


def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
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

    # vertices processed in ascending order of weight
    order = sorted(range(N), key=W.__getitem__)

    f = [0] * N  # f[i] : max operations starting with one piece on vertex i

    for v in order:
        cap = W[v] - 1  # weight budget for spawned pieces
        # neighbours that are strictly lighter
        items = [(W[u], f[u]) for u in adj[v] if W[u] < W[v]]
        best = knap_max_value(cap, items)
        f[v] = 1 + best  # 1 for the removal of the piece on v itself

    # final answer : sum over initial pieces
    ans = 0
    for i in range(N):
        ans += A[i] * f[i]

    print(ans)


if __name__ == "__main__":
    main()