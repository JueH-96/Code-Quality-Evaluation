import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]  # 1-based indexing

    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    W = list(map(int, sys.stdin.readline().split()))
    W = [0] + W  # W[1] is the weight of vertex 1

    A = list(map(int, sys.stdin.readline().split()))
    A = [0] + A  # A[1] is the count for vertex 1

    # Sort vertices in increasing order of their weights
    vertices = sorted(range(1, N + 1), key=lambda x: W[x])

    f = [0] * (N + 1)  # f[1..N] will store the values

    for x in vertices:
        current_W = W[x]
        cap = current_W - 1
        eligible = []
        for y in adj[x]:
            if W[y] < current_W:
                eligible.append(y)

        if cap < 0:
            f[x] = 1
            continue

        # Initialize DP array for this knapsack problem
        dp = [-float('inf')] * (cap + 1)
        dp[0] = 0

        for y in eligible:
            w = W[y]
            val = f[y]
            if w > cap:
                continue
            # Update the DP array in reverse to avoid overwriting values we need
            for j in range(cap, w - 1, -1):
                if dp[j - w] + val > dp[j]:
                    dp[j] = dp[j - w] + val

        max_val = max(dp)
        f[x] = 1 + max_val

    total = 0
    for i in range(1, N + 1):
        total += A[i] * f[i]

    print(total)

if __name__ == "__main__":
    main()