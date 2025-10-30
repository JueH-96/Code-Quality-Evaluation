# YOUR CODE HERE
n = int(input())
people = []
for i in range(n):
    a, b = map(int, input().split())
    people.append((a - 1, b))  # Convert to 0-indexed teams

total_strength = sum(b for _, b in people)

if total_strength % 3 != 0:
    print(-1)
else:
    target = total_strength // 3
    
    # DP approach with rolling array
    # dp[s1][s2] = minimum number of changes to achieve team 0 having strength s1 and team 1 having strength s2
    
    INF = float('inf')
    max_strength = total_strength + 1
    
    # Initialize DP
    dp = [[INF] * max_strength for _ in range(max_strength)]
    dp[0][0] = 0
    
    for i in range(n):
        original_team, strength = people[i]
        new_dp = [[INF] * max_strength for _ in range(max_strength)]
        
        for s1 in range(max_strength):
            for s2 in range(max_strength):
                if dp[s1][s2] == INF:
                    continue
                
                # Try assigning person i to each team
                for team in range(3):
                    if team == 0:
                        new_s1 = s1 + strength
                        new_s2 = s2
                    elif team == 1:
                        new_s1 = s1
                        new_s2 = s2 + strength
                    else:  # team == 2
                        new_s1 = s1
                        new_s2 = s2
                    
                    if new_s1 < max_strength and new_s2 < max_strength:
                        cost = 0 if team == original_team else 1
                        new_dp[new_s1][new_s2] = min(new_dp[new_s1][new_s2], dp[s1][s2] + cost)
        
        dp = new_dp
    
    result = dp[target][target]
    if result == INF:
        print(-1)
    else:
        print(result)