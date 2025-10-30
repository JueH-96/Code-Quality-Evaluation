n, t, m = map(int, input().split())
incompatible = []
for _ in range(m):
    a, b = map(int, input().split())
    incompatible.append((a-1, b-1))  # Convert to 0-indexed

teams = [-1] * n

def is_valid(player):
    for a, b in incompatible:
        if a == player:
            if teams[b] != -1 and teams[a] == teams[b]:
                return False
        elif b == player:
            if teams[a] != -1 and teams[a] == teams[b]:
                return False
    return True

def backtrack(player, max_team):
    if player == n:
        if max_team + 1 == t:
            return 1
        else:
            return 0
    
    count = 0
    # Try assigning to existing teams
    for team in range(max_team + 1):
        teams[player] = team
        if is_valid(player):
            count += backtrack(player + 1, max_team)
        teams[player] = -1
    
    # Try assigning to a new team (if possible)
    if max_team + 1 < t:
        teams[player] = max_team + 1
        if is_valid(player):
            count += backtrack(player + 1, max_team + 1)
        teams[player] = -1
    
    return count

print(backtrack(0, -1))