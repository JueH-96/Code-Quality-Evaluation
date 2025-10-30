def solve():
    n = int(input())
    courses = []
    for _ in range(n):
        x_i, y_i = map(int, input().split())
        courses.append({'type': x_i, 'tastiness': y_i})
    
    dp = [0, 0] # dp[0]: healthy stomach, dp[1]: upset stomach. Initialize for after course N.
    
    for i in range(n - 1, -1, -1):
        course = courses[i]
        x_i = course['type']
        y_i = course['tastiness']
        
        next_dp = [0, 0]
        
        # Calculate for healthy stomach state
        skip_healthy = dp[0]
        eat_healthy = 0
        if x_i == 0:
            eat_healthy = y_i + dp[0]
        else:
            eat_healthy = y_i + dp[1]
        next_dp[0] = max(skip_healthy, eat_healthy)
        
        # Calculate for upset stomach state
        skip_upset = dp[1]
        eat_upset = -float('inf')
        if x_i == 0:
            eat_upset = y_i + dp[0]
        else:
            eat_upset = -float('inf') # Cannot eat poisonous course in upset stomach
            
        if x_i == 0:
            next_dp[1] = max(skip_upset, eat_upset)
        else:
            next_dp[1] = skip_upset
            
        dp = next_dp
        
    print(dp[0])

if __name__ == '__main__':
    solve()