def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    presses = []
    for i in range(1, n + 1):
        parts = data[i].split()
        a = int(parts[0])
        s = parts[1].strip()
        presses.append((a, s))
    
    INF = 10**9
    dp = [[INF] * 101 for _ in range(101)]
    for l in range(1, 101):
        for r in range(1, 101):
            dp[l][r] = 0
            
    for a, s in presses:
        new_dp = [[INF] * 101 for _ in range(101)]
        for l in range(1, 101):
            for r in range(1, 101):
                if dp[l][r] == INF:
                    continue
                if s == 'L':
                    new_l = a
                    new_r = r
                    cost = abs(a - l)
                    total = dp[l][r] + cost
                    if total < new_dp[new_l][new_r]:
                        new_dp[new_l][new_r] = total
                else:
                    new_l = l
                    new_r = a
                    cost = abs(a - r)
                    total = dp[l][r] + cost
                    if total < new_dp[new_l][new_r]:
                        new_dp[new_l][new_r] = total
        dp = new_dp
        
    ans = INF
    for l in range(1, 101):
        for r in range(1, 101):
            if dp[l][r] < ans:
                ans = dp[l][r]
    print(ans)

if __name__ == '__main__':
    main()