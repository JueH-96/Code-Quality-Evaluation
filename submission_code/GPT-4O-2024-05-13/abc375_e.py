# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = []
    B = []
    
    for i in range(N):
        A.append(int(data[2 * i + 1]))
        B.append(int(data[2 * i + 2]))
    
    # Calculate the total strength of each team
    team_strength = [0, 0, 0]
    for i in range(N):
        team_strength[A[i] - 1] += B[i]
    
    total_strength = sum(team_strength)
    
    # If the total strength is not divisible by 3, it's impossible to balance
    if total_strength % 3 != 0:
        print(-1)
        return
    
    target_strength = total_strength // 3
    
    # If all teams already have the target strength
    if team_strength[0] == target_strength and team_strength[1] == target_strength and team_strength[2] == target_strength:
        print(0)
        return
    
    # Dynamic programming approach to find the minimum number of switches
    from collections import defaultdict
    dp = defaultdict(lambda: float('inf'))
    dp[(team_strength[0], team_strength[1], team_strength[2])] = 0
    
    for i in range(N):
        new_dp = defaultdict(lambda: float('inf'))
        for (s1, s2, s3), switches in dp.items():
            if A[i] == 1:
                new_dp[(s1 - B[i], s2 + B[i], s3)] = min(new_dp[(s1 - B[i], s2 + B[i], s3)], switches + 1)
                new_dp[(s1 - B[i], s2, s3 + B[i])] = min(new_dp[(s1 - B[i], s2, s3 + B[i])], switches + 1)
            elif A[i] == 2:
                new_dp[(s1 + B[i], s2 - B[i], s3)] = min(new_dp[(s1 + B[i], s2 - B[i], s3)], switches + 1)
                new_dp[(s1, s2 - B[i], s3 + B[i])] = min(new_dp[(s1, s2 - B[i], s3 + B[i])], switches + 1)
            elif A[i] == 3:
                new_dp[(s1 + B[i], s2, s3 - B[i])] = min(new_dp[(s1 + B[i], s2, s3 - B[i])], switches + 1)
                new_dp[(s1, s2 + B[i], s3 - B[i])] = min(new_dp[(s1, s2 + B[i], s3 - B[i])], switches + 1)
        
        for key in new_dp:
            dp[key] = min(dp[key], new_dp[key])
    
    result = dp[(target_strength, target_strength, target_strength)]
    if result == float('inf'):
        print(-1)
    else:
        print(result)