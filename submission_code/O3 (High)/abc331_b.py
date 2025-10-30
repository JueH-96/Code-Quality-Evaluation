import sys


def minimum_cost(N: int, S: int, M: int, L: int) -> int:
    """
    Return the minimum amount of money needed to buy at least N eggs
    using packs of 6 (price S), 8 (price M) and 12 (price L).
    """
    # An optimal solution never exceeds N + 11 eggs (see reasoning in analysis),
    # so we only need to explore up to that many eggs.
    MAX_EGGS = N + 12          # inclusive upper bound (N + 11 would also work)
    INF = 10 ** 18

    # dp[i] = minimal cost to obtain exactly i eggs
    dp = [INF] * (MAX_EGGS + 1)
    dp[0] = 0

    for i in range(MAX_EGGS + 1):
        if dp[i] == INF:
            continue
        for size, cost in ((6, S), (8, M), (12, L)):
            nxt = i + size
            if nxt <= MAX_EGGS and dp[nxt] > dp[i] + cost:
                dp[nxt] = dp[i] + cost

    return min(dp[N:])   # minimal cost for any amount ≥ N (within bound)


def main() -> None:
    N, S, M, L = map(int, sys.stdin.read().split())
    print(minimum_cost(N, S, M, L))


if __name__ == "__main__":
    main()