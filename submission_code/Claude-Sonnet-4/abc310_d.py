def solve():
    N, T, M = map(int, input().split())
    
    incompatible = []
    for _ in range(M):
        a, b = map(int, input().split())
        incompatible.append((a-1, b-1))  # Convert to 0-indexed
    
    def generate_assignments(player, assignment, teams_used):
        if player == N:
            # Check if exactly T teams are used
            if len(teams_used) != T:
                return 0
            
            # Check incompatibility constraints
            for a, b in incompatible:
                if assignment[a] == assignment[b]:
                    return 0
            
            return 1
        
        count = 0
        # Try assigning current player to existing teams
        for team in teams_used:
            assignment[player] = team
            count += generate_assignments(player + 1, assignment, teams_used)
        
        # Try assigning current player to a new team (if we haven't used T teams yet)
        if len(teams_used) < T:
            new_team = len(teams_used)
            assignment[player] = new_team
            count += generate_assignments(player + 1, assignment, teams_used | {new_team})
        
        return count
    
    assignment = [0] * N
    result = generate_assignments(0, assignment, set())
    print(result)

solve()