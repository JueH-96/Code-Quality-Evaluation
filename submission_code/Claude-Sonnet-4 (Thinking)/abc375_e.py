n = int(input())
people = []
for _ in range(n):
    a, b = map(int, input().split())
    people.append((a, b))

total_strength = sum(b for a, b in people)

if total_strength % 3 != 0:
    print(-1)
else:
    target = total_strength // 3
    
    # dp[i][s1][s2] = minimum switches to assign first i people 
    # such that team 1 has strength s1, team 2 has strength s2
    dp = [[[float('inf')] * (target + 1) for _ in range(target + 1)] for _ in range(n + 1)]
    
    dp[0][0][0] = 0
    
    for i in range(n):
        original_team, strength = people[i]
        for s1 in range(target + 1):
            for s2 in range(target + 1):
                if dp[i][s1][s2] == float('inf'):
                    continue
                
                # Assign person i to team 1
                if s1 + strength <= target:
                    cost = 0 if original_team == 1 else 1
                    dp[i + 1][s1 + strength][s2] = min(dp[i + 1][s1 + strength][s2], dp[i][s1][s2] + cost)
                
                # Assign person i to team 2
                if s2 + strength <= target:
                    cost = 0 if original_team == 2 else 1
                    dp[i + 1][s1][s2 + strength] = min(dp[i + 1][s1][s2 + strength], dp[i][s1][s2] + cost)
                
                # Assign person i to team 3
                cost = 0 if original_team == 3 else 1
                dp[i + 1][s1][s2] = min(dp[i + 1][s1][s2], dp[i][s1][s2] + cost)
    
    result = dp[n][target][target]
    print(result if result != float('inf') else -1)