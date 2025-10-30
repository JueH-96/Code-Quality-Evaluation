import sys
import threading

def main():
    data = sys.stdin.read().strip().split()
    N, S, M, L = map(int, data)

    # We want at least N eggs. We can overshoot by up to 12 (the largest pack size).
    max_eggs = N + 12

    # dp[i] = minimum cost to get exactly i eggs
    INF = 10**18
    dp = [INF] * (max_eggs + 1)
    dp[0] = 0

    packs = [(6, S), (8, M), (12, L)]
    for size, cost in packs:
        for i in range(size, max_eggs + 1):
            dp[i] = min(dp[i], dp[i - size] + cost)

    # answer is the minimum cost among dp[N], dp[N+1], ..., dp[max_eggs]
    answer = min(dp[N:])
    print(answer)

if __name__ == "__main__":
    main()