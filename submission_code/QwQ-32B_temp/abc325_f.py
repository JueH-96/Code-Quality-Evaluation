import sys

def main():
    import sys
    N = int(sys.stdin.readline())
    D = list(map(int, sys.stdin.readline().split()))
    L1, C1, K1 = map(int, sys.stdin.readline().split())
    L2, C2, K2 = map(int, sys.stdin.readline().split())

    sections = []
    for d in D:
        a_max = min((d + L1 - 1) // L1, K1)
        candidates = {0, 1, 2, a_max - 1, a_max}
        candidates = [a for a in candidates if 0 <= a <= a_max]
        options = []
        seen = set()
        for a in candidates:
            rem = d - a * L1
            if rem <= 0:
                b = 0
            else:
                b = (rem + L2 - 1) // L2
            cost = a * C1 + b * C2
            if (a, b) not in seen:
                seen.add((a, b))
                options.append((a, b, cost))
        sections.append(options)

    INF = float('inf')
    dp = [(INF, 0) for _ in range(K1 + 1)]
    dp[0] = (0, 0)

    for section in sections:
        new_dp = [(INF, 0) for _ in range(K1 + 1)]
        for a in range(K1 + 1):
            current_cost, current_b = dp[a]
            if current_cost == INF:
                continue
            for da, db, c in section:
                new_a = a + da
                new_b = current_b + db
                new_cost = current_cost + c
                if new_a > K1 or new_b > K2:
                    continue
                if (new_cost < new_dp[new_a][0]) or (new_cost == new_dp[new_a][0] and new_b < new_dp[new_a][1]):
                    new_dp[new_a] = (new_cost, new_b)
        dp = new_dp

    min_cost = INF
    for a in range(K1 + 1):
        cost, b = dp[a]
        if cost < INF and b <= K2:
            if cost < min_cost:
                min_cost = cost

    print(-1 if min_cost == INF else min_cost)

if __name__ == "__main__":
    main()