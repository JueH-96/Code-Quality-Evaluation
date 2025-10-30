import sys

def main():
    input = sys.stdin.readline
    T = input().strip()
    if not T:
        print(-1)
        return
    N = int(input())
    bags = []
    for _ in range(N):
        parts = input().split()
        A = int(parts[0])
        strs = parts[1:]
        bags.append(strs)

    M = len(T)
    INF = 10**9
    # dp[pos] = minimum cost to build T[:pos] using processed bags so far
    dp = [INF] * (M + 1)
    dp[0] = 0

    for strs in bags:
        # start dp_next with skip-transitions (i.e., carry over dp)
        dp_next = dp[:]
        for pos in range(M + 1):
            cost = dp[pos]
            if cost == INF:
                continue
            # try picking one string from this bag
            for s in strs:
                L = len(s)
                if pos + L <= M and T.startswith(s, pos):
                    if dp_next[pos + L] > cost + 1:
                        dp_next[pos + L] = cost + 1
        dp = dp_next

    ans = dp[M]
    print(ans if ans < INF else -1)

if __name__ == "__main__":
    main()