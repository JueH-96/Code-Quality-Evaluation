def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, S, M, L = map(int, data)
    
    # We'll use a dynamic programming approach.
    # dp[i] will represent the minimum cost to buy exactly i eggs (if possible).
    # However, because we can overshoot the number of eggs (buy more than N),
    # we'll compute dp values up to N + 12 (the largest pack size).
    # Then, our answer is the minimum cost for all dp[i] with i >= N.
    
    INF = 10**15
    max_eggs = N + 12  # a bit more to allow overshoot
    dp = [INF] * (max_eggs + 1)
    dp[0] = 0  # cost of 0 for 0 eggs
    
    for i in range(max_eggs + 1):
        if dp[i] == INF:
            continue
        if i + 6 <= max_eggs:
            dp[i + 6] = min(dp[i + 6], dp[i] + S)
        if i + 8 <= max_eggs:
            dp[i + 8] = min(dp[i + 8], dp[i] + M)
        if i + 12 <= max_eggs:
            dp[i + 12] = min(dp[i + 12], dp[i] + L)
    
    # The answer is the minimum cost among dp[i] for i >= N
    answer = min(dp[N:])
    print(answer)

# Let's call solve() to handle the process.
def main():
    solve()

if __name__ == "__main__":
    main()