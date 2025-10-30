def solve():
    n = int(input())
    teams = []
    strengths = []
    for _ in range(n):
        a, b = map(int, input().split())
        teams.append(a)
        strengths.append(b)

    total_strength = sum(strengths)
    if total_strength % 3 != 0:
        print(-1)
        return

    target_strength = total_strength // 3
    
    initial_team_strengths = [0] * 3
    for i in range(n):
        initial_team_strengths[teams[i]-1] += strengths[i]

    if all(s == target_strength for s in initial_team_strengths):
        print(0)
        return

    min_switches = float('inf')
    
    
    for i in range(1 << (2 * n)):
        current_teams = list(teams)
        num_switches = 0
        
        for j in range(n):
            team_bits = (i >> (2 * j)) & 3
            if team_bits == 0:
                new_team = 1
            elif team_bits == 1:
                new_team = 2
            elif team_bits == 2:
                new_team = 3
            else:
                new_team = current_teams[j]
            
            if new_team != current_teams[j]:
                num_switches += 1
                current_teams[j] = new_team

        current_team_strengths = [0] * 3
        for j in range(n):
            current_team_strengths[current_teams[j] - 1] += strengths[j]

        if all(s == target_strength for s in current_team_strengths):
            min_switches = min(min_switches, num_switches)

    if min_switches == float('inf'):
        print(-1)
    else:
        print(min_switches)

solve()