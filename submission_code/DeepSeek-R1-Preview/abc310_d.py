n, t, m = map(int, input().split())

incompatibles = [[] for _ in range(n)]
for _ in range(m):
    a, b = map(int, input().split())
    a -= 1
    b -= 1
    incompatibles[a].append(b)
    incompatibles[b].append(a)

def backtrack(index, team_assignment, max_team):
    if index == n:
        if max_team + 1 == t:
            return 1
        else:
            return 0
    total = 0
    current_player = index
    for team in range(0, max_team + 2):
        valid = True
        for v in incompatibles[current_player]:
            if v < index:  # Only check already assigned players
                if team_assignment[v] == team:
                    valid = False
                    break
        if not valid:
            continue
        new_team_assignment = team_assignment.copy()
        new_team_assignment.append(team)
        new_max_team = max(max_team, team)
        if new_max_team >= t:
            continue
        total += backtrack(index + 1, new_team_assignment, new_max_team)
    return total

result = backtrack(0, [], -1)
print(result)