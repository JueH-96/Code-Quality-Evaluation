import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    arr = []
    total_sum = 0
    for i in range(1, n + 1):
        parts = data[i].split()
        a = int(parts[0])
        b = int(parts[1])
        arr.append((a, b))
        total_sum += b
    
    if total_sum % 3 != 0:
        print(-1)
        return
        
    target = total_sum // 3
    INF = 10**9
    dp = [[INF] * (target + 1) for _ in range(target + 1)]
    dp[0][0] = 0
    
    for a, b in arr:
        new_dp = [[INF] * (target + 1) for _ in range(target + 1)]
        for s1 in range(target + 1):
            for s2 in range(target + 1):
                if dp[s1][s2] == INF:
                    continue
                # Option 1: assign to team1
                ns1 = s1 + b
                ns2 = s2
                if ns1 <= target and ns2 <= target:
                    cost = dp[s1][s2] + (0 if a == 1 else 1)
                    if cost < new_dp[ns1][ns2]:
                        new_dp[ns1][ns2] = cost
                # Option 2: assign to team2
                ns1 = s1
                ns2 = s2 + b
                if ns1 <= target and ns2 <= target:
                    cost = dp[s1][s2] + (0 if a == 2 else 1)
                    if cost < new_dp[ns1][ns2]:
                        new_dp[ns1][ns2] = cost
                # Option 3: assign to team3
                cost = dp[s1][s2] + (0 if a == 3 else 1)
                if cost < new_dp[s1][s2]:
                    new_dp[s1][s2] = cost
        dp = new_dp
        
    if dp[target][target] < INF:
        print(dp[target][target])
    else:
        print(-1)

if __name__ == '__main__':
    main()