from collections import defaultdict

def solve():
    N = int(input())
    team_strengths = [0] * 3
    team_members = defaultdict(list)
    
    for i in range(N):
        a, b = map(int, input().split())
        team_strengths[a-1] += b
        team_members[a-1].append(i)
    
    if len(set(team_strengths)) == 1:
        return 0
    
    avg_strength = sum(team_strengths) // 3
    
    min_switches = float('inf')
    for i in range(3):
        for j in range(i+1, 3):
            diff = abs(team_strengths[i] - team_strengths[j])
            if diff % 2 == 0 and diff // 2 <= len(team_members[i]) and diff // 2 <= len(team_members[j]):
                switches = diff // 2
                min_switches = min(min_switches, switches)
    
    return min_switches if min_switches != float('inf') else -1

print(solve())