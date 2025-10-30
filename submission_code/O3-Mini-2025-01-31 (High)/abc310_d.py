def main():
    import sys
    
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    T = int(data[1])
    M = int(data[2])
    
    # Build a conflict (incompatibility) matrix for players (0-indexed)
    conflicts = [[False]*N for _ in range(N)]
    idx = 3
    for _ in range(M):
        a = int(data[idx]) - 1
        b = int(data[idx+1]) - 1
        idx += 2
        conflicts[a][b] = True
        conflicts[b][a] = True
    
    # We'll form partitions of players into exactly T teams.
    # Two divisions are considered the same if the "teammate relation" is the same.
    # We use a canonical backtracking: assign players sequentially,
    # allowing assignment to an already used team or (if available) to a new team.
    # This ensures we don't count labellings that are isomorphic.
    
    groups = [-1] * N  # groups[i] is the team number assigned to player i (0-indexed team number)
    ans = 0

    # backtrack will try to assign players starting from 'player' index,
    # 'used' is the number of teams that have been used so far.
    def backtrack(player, used):
        nonlocal ans
        if player == N:
            # Only count partitions that use exactly T teams.
            if used == T:
                ans += 1
            return
            
        # Try assigning player to each of the already used teams.
        for team in range(used):
            valid = True
            # Check that no previously assigned player in the same team conflicts with the current player.
            for j in range(player):
                if groups[j] == team and conflicts[player][j]:
                    valid = False
                    break
            if valid:
                groups[player] = team
                backtrack(player+1, used)
                groups[player] = -1
        
        # If possible, assign player to a new team.
        if used < T:
            groups[player] = used
            backtrack(player+1, used+1)
            groups[player] = -1

    # Begin by assigning the first player to team 0 to enforce a canonical representation.
    groups[0] = 0
    backtrack(1, 1)
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()