# YOUR CODE HERE
def solve():
    N, T, M = map(int, input().split())
    
    # Read incompatible pairs
    incompatible = []
    for _ in range(M):
        a, b = map(int, input().split())
        incompatible.append((a-1, b-1))  # Convert to 0-indexed
    
    # Check if two players can be in the same team
    def can_be_together(team_mask, player):
        for a, b in incompatible:
            if a == player:
                if (team_mask >> b) & 1:
                    return False
            elif b == player:
                if (team_mask >> a) & 1:
                    return False
        return True
    
    # DP to count valid partitions
    # dp[mask][k] = number of ways to partition players in mask into k teams
    dp = {}
    
    def count_partitions(mask, k):
        if k == 0:
            return 1 if mask == 0 else 0
        if mask == 0:
            return 0
        if bin(mask).count('1') < k:
            return 0
            
        if (mask, k) in dp:
            return dp[(mask, k)]
        
        # Find the lowest set bit (first player to assign)
        first_player = -1
        for i in range(N):
            if (mask >> i) & 1:
                first_player = i
                break
        
        result = 0
        
        # Try all possible subsets that include first_player
        submask = mask
        while submask > 0:
            if (submask >> first_player) & 1:  # Must include first_player
                # Check if this subset forms a valid team
                valid = True
                team_mask = submask
                
                # Check each player in the team
                for i in range(N):
                    if (team_mask >> i) & 1:
                        if not can_be_together(team_mask, i):
                            valid = False
                            break
                
                if valid:
                    # Remove this team and count remaining partitions
                    remaining = mask ^ submask
                    result += count_partitions(remaining, k - 1)
            
            # Generate next submask
            submask = (submask - 1) & mask
        
        dp[(mask, k)] = result
        return result
    
    # Start with all players (full mask)
    full_mask = (1 << N) - 1
    print(count_partitions(full_mask, T))

solve()