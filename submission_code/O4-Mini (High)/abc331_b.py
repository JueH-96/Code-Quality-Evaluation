def main():
    import sys
    data = sys.stdin.read().split()
    N, S, M, L = map(int, data)
    # We do an unbounded knapsack DP up to N + max_pack_size (12),
    # then take the minimum cost among dp[e] for e >= N.
    max_eggs = N + 12
    INF = 10**18
    dp = [INF] * (max_eggs + 1)
    dp[0] = 0

    for e in range(1, max_eggs + 1):
        if e >= 6:
            dp[e] = min(dp[e], dp[e - 6] + S)
        if e >= 8:
            dp[e] = min(dp[e], dp[e - 8] + M)
        if e >= 12:
            dp[e] = min(dp[e], dp[e - 12] + L)

    answer = min(dp[N:])  # minimum cost to get at least N eggs
    print(answer)

# Call main to execute
main()