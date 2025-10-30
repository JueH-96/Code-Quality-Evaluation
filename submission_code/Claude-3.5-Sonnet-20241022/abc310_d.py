def solve():
    N, T, M = map(int, input().split())
    
    # Create adjacency list for incompatible pairs
    incompatible = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b = map(int, input().split())
        incompatible[a].append(b)
        incompatible[b].append(a)
    
    # dp[mask][last] = number of ways to assign teams to players in mask,
    # where last is the last team number used (1-based)
    dp = {}
    
    def check_valid(mask, team):
        # Check if any incompatible players are in same team
        players = []
        for i in range(1, N+1):
            if team[i] != 0 and ((1 << (i-1)) & mask):
                for j in incompatible[i]:
                    if team[j] == team[i] and ((1 << (j-1)) & mask):
                        return False
        return True
    
    def solve_dp(mask, last, team):
        if mask == 0:
            return 1 if last > 0 else 0
            
        if (mask, last) in dp:
            return dp[(mask, last)]
            
        result = 0
        # Try assigning next player to teams 1 to min(last+1, T)
        next_player = 0
        for i in range(N):
            if mask & (1 << i):
                next_player = i + 1
                break
                
        max_team = min(last + 1, T)
        for t in range(1, max_team + 1):
            team[next_player] = t
            if check_valid(mask, team):
                result += solve_dp(mask ^ (1 << (next_player-1)), max(last, t), team)
            team[next_player] = 0
            
        dp[(mask, last)] = result
        return result
    
    team = [0] * (N+1)
    initial_mask = (1 << N) - 1
    ans = solve_dp(initial_mask, 0, team)
    print(ans)

solve()