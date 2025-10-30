import sys

def main():
    import sys
    N = int(sys.stdin.readline())
    D = list(map(int, sys.stdin.readline().split()))
    L1, C1, K1 = map(int, sys.stdin.readline().split())
    L2, C2, K2 = map(int, sys.stdin.readline().split())
    
    sections = []
    for d in D:
        pairs = []
        a_max = min((d + L1 - 1) // L1, K1)
        for a_i in range(0, a_max + 1):
            remaining = d - a_i * L1
            if remaining <= 0:
                b_i = 0
            else:
                b_i = (remaining + L2 - 1) // L2
            cost = a_i * C1 + b_i * C2
            pairs.append((a_i, b_i, cost))
        sections.append(pairs)
    
    INF = float('inf')
    dp = [[INF] * (K2 + 1) for _ in range(K1 + 1)]
    dp[0][0] = 0
    
    for section in sections:
        new_dp = [[INF] * (K2 + 1) for _ in range(K1 + 1)]
        for a in range(K1 + 1):
            for b in range(K2 + 1):
                if dp[a][b] == INF:
                    continue
                for (a_i, b_i, cost_i) in section:
                    new_a = a + a_i
                    new_b = b + b_i
                    if new_a > K1 or new_b > K2:
                        continue
                    if new_dp[new_a][new_b] > dp[a][b] + cost_i:
                        new_dp[new_a][new_b] = dp[a][b] + cost_i
        dp = new_dp
    
    min_cost = INF
    for a in range(K1 + 1):
        for b in range(K2 + 1):
            if a <= K1 and b <= K2 and dp[a][b] < min_cost:
                min_cost = dp[a][b]
    
    if min_cost == INF:
        print(-1)
    else:
        print(min_cost)

if __name__ == '__main__':
    main()