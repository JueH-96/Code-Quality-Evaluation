def solve():
    n = int(input())
    s_list = [input() for _ in range(n)]
    
    results = []
    for k in range(1, n + 1):
        target_string = s_list[k-1]
        min_cost = len(target_string) # Cost to make it empty string
        
        previous_strings = s_list[:k-1]
        if not previous_strings:
            results.append(min_cost)
            continue
            
        for s_prime in previous_strings:
            n1 = len(target_string)
            n2 = len(s_prime)
            dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
            for i in range(n1 + 1):
                dp[i][0] = i
            for j in range(n2 + 1):
                dp[0][j] = j
                
            for i in range(1, n1 + 1):
                for j in range(1, n2 + 1):
                    if target_string[i-1] == s_prime[j-1]:
                        dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1])
                    else:
                        dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1)
                        
            min_cost = min(min_cost, dp[n1][n2])
            
        results.append(min_cost)
        
    for cost in results:
        print(cost)

if __name__ == '__main__':
    solve()