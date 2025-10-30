def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N, S, M, L = map(int, data)
    # We'll compute dp[i] = minimum cost to get exactly i eggs (or INF if impossible)
    # Then answer = min(dp[i] for i >= N up to N+12)
    INF = 10**18
    max_pack = 12
    max_eggs = N + max_pack
    dp = [INF] * (max_eggs + 1)
    dp[0] = 0
    packs = [(6, S), (8, M), (12, L)]
    for i in range(1, max_eggs + 1):
        best = INF
        for eggs, cost in packs:
            if i >= eggs:
                c = dp[i - eggs] + cost
                if c < best:
                    best = c
        dp[i] = best
    # Now find the minimum cost for at least N eggs
    ans = min(dp[N:])
    print(ans)

if __name__ == "__main__":
    main()