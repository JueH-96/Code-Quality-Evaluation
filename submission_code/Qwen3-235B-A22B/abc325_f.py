import sys

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    D = list(map(int, input[ptr:ptr+N]))
    ptr += N
    L1 = int(input[ptr])
    C1 = int(input[ptr+1])
    K1 = int(input[ptr+2])
    ptr +=3
    L2 = int(input[ptr])
    C2 = int(input[ptr+1])
    K2 = int(input[ptr+2])
    ptr +=3
    
    INF = float('inf')
    dp = [[INF] * (K2 + 1) for _ in range(K1 + 1)]
    dp[0][0] = 0
    
    for d in D:
        new_dp = [[INF] * (K2 + 1) for _ in range(K1 + 1)]
        pairs = []
        max_x = (d + L1 - 1) // L1
        max_x = min(max_x, K1)
        
        for x in range(0, max_x + 1):
            rem = d - x * L1
            if rem <= 0:
                y = 0
            else:
                y = (rem + L2 - 1) // L2
            pairs.append((x, y))
        
        for a in range(K1 + 1):
            if dp[a][0] == INF:
                for b in range(K2 + 1):
                    if dp[a][b] < INF:
                        break
                else:
                    continue
            for b in range(K2 + 1):
                if dp[a][b] == INF:
                    continue
                cost_ab = dp[a][b]
                for (x, y) in pairs:
                    na = a + x
                    if na > K1:
                        continue
                    rem_y = K2 + 1 - y
                    nb = b + y
                    if nb > K2:
                        continue
                    total_cost = cost_ab + C1 * x + C2 * y
                    if new_dp[na][nb] > total_cost:
                        new_dp[na][nb] = total_cost
        
        dp = new_dp
    
    res = INF
    for a in range(K1 + 1):
        for b in range(K2 + 1):
            if dp[a][b] < res:
                res = dp[a][b]
    
    print(-1 if res == INF else res)

if __name__ == "__main__":
    main()