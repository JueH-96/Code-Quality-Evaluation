def solve():
    n, x = map(int, input().split())
    machines = []
    for _ in range(n):
        machines.append(list(map(int, input().split())))

    def check(capacity):
        dp = [float('inf')] * (x + 1)
        dp[0] = 0
        for i in range(n):
            new_dp = [float('inf')] * (x + 1)
            for j in range(x + 1):
                if dp[j] == float('inf'):
                    continue
                
                for k in range(x // machines[i][1] + 1):
                    if j + k * machines[i][1] <= x:
                        new_dp[j + k * machines[i][1]] = min(new_dp[j + k * machines[i][1]], dp[j] + k * machines[i][0])
                
                for k in range(x // machines[i][3] + 1):
                    if j + k * machines[i][3] <= x:
                        new_dp[j + k * machines[i][3]] = min(new_dp[j + k * machines[i][3]], dp[j] + k * machines[i][2])
            dp = new_dp
        
        
        ans = 0
        for cost in range(x+1):
            if dp[cost] >= capacity:
                ans = capacity
                break
        return ans
    
    left, right = 0, x * 100 + 1
    ans = 0
    while left <= right:
        mid = (left + right) // 2
        if check(mid) == mid:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1

    print(ans)

solve()