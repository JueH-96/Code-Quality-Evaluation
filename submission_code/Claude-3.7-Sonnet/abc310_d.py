def main():
    line = input().split()
    N = int(line[0])  # Number of players
    T = int(line[1])  # Number of teams
    M = int(line[2])  # Number of incompatible pairs
    
    # Create an adjacency matrix for the incompatibility graph
    graph = [[0] * (N + 1) for _ in range(N + 1)]
    for _ in range(M):
        a, b = map(int, input().split())
        graph[a][b] = graph[b][a] = 1
    
    # Initialize team assignments
    team = [0] * (N + 1)
    
    # Backtracking function
    def backtrack(player, max_team_used):
        if player > N:
            return 1 if max_team_used == T else 0
        
        result = 0
        
        # Try assigning to an existing team
        for t in range(1, max_team_used + 1):
            valid = True
            for prev_player in range(1, player):
                if graph[player][prev_player] and team[prev_player] == t:
                    valid = False
                    break
            if valid:
                team[player] = t
                result += backtrack(player + 1, max_team_used)
        
        # Try assigning to a new team
        if max_team_used < T:
            team[player] = max_team_used + 1
            result += backtrack(player + 1, max_team_used + 1)
        
        return result
    
    # Start backtracking from player 1 with no teams used yet
    result = backtrack(1, 0)
    print(result)

if __name__ == "__main__":
    main()