from itertools import combinations

def count_valid_divisions(N, T, M, incompatible_pairs):
    # Generate all possible ways to assign players to teams
    # Each player can be assigned to any of the T teams
    # We need to count the number of valid assignments where no incompatible pair is in the same team
    
    # Preprocess incompatible pairs for quick lookup
    incompatible = [[] for _ in range(N+1)]
    for a, b in incompatible_pairs:
        incompatible[a].append(b)
        incompatible[b].append(a)
    
    # We will use a recursive approach to assign players to teams
    # and ensure that no incompatible pairs are in the same team
    
    # Initialize the count of valid divisions
    count = 0
    
    # We will represent the team assignment as a list where the i-th element is the team of the i-th player
    # We will use a recursive function to assign teams to players
    
    def backtrack(player, team_assignment):
        nonlocal count
        if player > N:
            # All players are assigned, check if the number of teams used is exactly T
            if len(set(team_assignment[1:])) == T:
                count += 1
            return
        # Try assigning the current player to each possible team
        for team in range(1, T+1):
            # Check if assigning this team to the current player is valid
            valid = True
            for neighbor in incompatible[player]:
                if team_assignment[neighbor] == team:
                    valid = False
                    break
            if valid:
                team_assignment[player] = team
                backtrack(player+1, team_assignment)
                team_assignment[player] = 0
    
    # Initialize the team assignment list
    team_assignment = [0] * (N+1)
    backtrack(1, team_assignment)
    
    return count

# Read input
N, T, M = map(int, input().split())
incompatible_pairs = [tuple(map(int, input().split())) for _ in range(M)]

# Compute the number of valid divisions
result = count_valid_divisions(N, T, M, incompatible_pairs)

# Print the result
print(result)