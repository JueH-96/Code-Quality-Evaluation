def get_team_strengths(N, teams, strengths):
    team_strengths = [0] * 4  # 1-based indexing
    for i in range(N):
        team_strengths[teams[i]] += strengths[i]
    return team_strengths[1:]  # return only teams 1,2,3

def can_make_equal(N, teams, strengths):
    total_strength = sum(strengths)
    if total_strength % 3 != 0:
        return -1
    
    target = total_strength // 3
    current = get_team_strengths(N, teams, strengths)
    
    # Try all possible movements of people
    min_moves = float('inf')
    
    # For each person, try moving them to each other team
    for mask in range(1 << N):
        moves = bin(mask).count('1')
        if moves >= min_moves:
            continue
            
        new_teams = teams.copy()
        # Apply moves based on binary mask
        for i in range(N):
            if mask & (1 << i):
                # Try both other teams
                for new_team in [1, 2, 3]:
                    if new_team != teams[i]:
                        new_teams[i] = new_team
                        new_strengths = get_team_strengths(N, new_teams, strengths)
                        if new_strengths[0] == new_strengths[1] == new_strengths[2]:
                            min_moves = min(min_moves, moves)
                        new_teams[i] = teams[i]
    
    return min_moves if min_moves != float('inf') else -1

# Read input
N = int(input())
teams = []
strengths = []

for _ in range(N):
    a, b = map(int, input().split())
    teams.append(a)
    strengths.append(b)

# Print result
print(can_make_equal(N, teams, strengths))