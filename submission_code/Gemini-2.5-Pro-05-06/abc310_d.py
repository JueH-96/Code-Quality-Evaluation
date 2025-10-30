import sys

def solve():
    N, T, M = map(int, sys.stdin.readline().split())

    incompatible = [[False for _ in range(N)] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        # Adjust to 0-indexed
        u -= 1
        v -= 1
        incompatible[u][v] = True
        incompatible[v][u] = True # Graph is undirected, so mark both ways

    # player_team[i] stores the team index for player i
    # Team indices will be 0, 1, ..., T-1
    player_team = [-1] * N 
    
    # Using a list of size 1 to pass integer by reference effectively
    count_arr = [0]

    def dfs(k, teams_count):
        # k: current player_idx to assign a team (from 0 to N-1)
        # teams_count: number of distinct teams currently formed by players 0...k-1
        
        if k == N:
            if teams_count == T:
                count_arr[0] += 1
            return

        if teams_count + (N - k) < T:
            return
        
        # Option 1: Assign player k to one of the existing 'teams_count' teams
        # These teams are conventionally numbered 0, 1, ..., teams_count-1
        for t_idx in range(teams_count):
            player_team[k] = t_idx
            
            possible_to_assign_to_this_team = True
            for prev_player_idx in range(k):
                if player_team[prev_player_idx] == t_idx: 
                    if incompatible[prev_player_idx][k]:
                        possible_to_assign_to_this_team = False
                        break 
            
            if possible_to_assign_to_this_team:
                dfs(k + 1, teams_count)
        
        # Option 2: Assign player k to a new team
        if teams_count < T:
            player_team[k] = teams_count 
            
            # No incompatibility check with players 0...k-1 is needed here,
            # as player k is in a new team, different from all their teams.
            dfs(k + 1, teams_count + 1)
        
    dfs(0, 0)
    
    print(count_arr[0])

solve()