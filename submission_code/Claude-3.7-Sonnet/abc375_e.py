def min_people_to_switch():
    n = int(input())
    teams = []
    strengths = []
    
    for _ in range(n):
        a, b = map(int, input().split())
        teams.append(a)
        strengths.append(b)
    
    total_strength = sum(strengths)
    
    # If total strength is not divisible by 3, it's impossible
    if total_strength % 3 != 0:
        return -1
    
    target_strength = total_strength // 3
    
    # DP approach: dp[(s1, s2, s3)] = minimum number of people who need to switch
    # to have teams with strengths s1, s2, and s3
    dp = {}
    dp[(0, 0, 0)] = 0
    
    for i in range(n):
        new_dp = {}
        for (s1, s2, s3), switches in dp.items():
            # Try adding person i to team 1
            if s1 + strengths[i] <= target_strength:
                cost = 0 if teams[i] == 1 else 1
                key = (s1 + strengths[i], s2, s3)
                if key not in new_dp or new_dp[key] > switches + cost:
                    new_dp[key] = switches + cost
            
            # Try adding person i to team 2
            if s2 + strengths[i] <= target_strength:
                cost = 0 if teams[i] == 2 else 1
                key = (s1, s2 + strengths[i], s3)
                if key not in new_dp or new_dp[key] > switches + cost:
                    new_dp[key] = switches + cost
            
            # Try adding person i to team 3
            if s3 + strengths[i] <= target_strength:
                cost = 0 if teams[i] == 3 else 1
                key = (s1, s2, s3 + strengths[i])
                if key not in new_dp or new_dp[key] > switches + cost:
                    new_dp[key] = switches + cost
        
        dp = new_dp
    
    # Check if it's possible to have equal strength teams
    return dp.get((target_strength, target_strength, target_strength), -1)

print(min_people_to_switch())