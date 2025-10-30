MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx += 1
    M = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1

    edges = [[] for _ in range(N + 1)]  # edges[u] is list of x's with x -> u
    for _ in range(M):
        x = int(data[idx]); idx += 1
        y = int(data[idx]); idx += 1
        edges[y].append(x)

    # Precompute u_prev for each u (1-based)
    u_prev = [0] * (N + 1)
    for u in range(1, N + 1):
        if u == 1:
            u_prev[u] = N
        else:
            u_prev[u] = u - 1

    # Initialize dp_prev
    dp_prev = [0] * (N + 1)
    dp_prev[1] = 1

    for _ in range(K):
        # Compute shifted_dp
        shifted_dp = [0] * (N + 1)
        for u in range(1, N + 1):
            shifted_dp[u] = dp_prev[u_prev[u]]
        # Apply extra edges
        for u in range(1, N + 1):
            if edges[u]:
                total = 0
                for x in edges[u]:
                    total = (total + dp_prev[x]) % MOD
                shifted_dp[u] = (shifted_dp[u] + total) % MOD
        dp_prev = shifted_dp

    print(dp_prev[1] % MOD)

if __name__ == '__main__':
    main()