# YOUR CODE HERE
MOD = 998244353

def modinv(a, m=MOD):
    return pow(a, m - 2, m)

def solve():
    N = int(input())
    
    # Initialize: each player is in their own team
    teams = [{i} for i in range(N)]  # teams[i] contains set of players in team i
    player_to_team = list(range(N))  # player_to_team[i] = team index of player i
    
    # Expected wins for each player (as fractions)
    # We'll store numerator and denominator separately to avoid precision issues
    expected_wins = [0] * N
    
    for match in range(N - 1):
        p, q = map(int, input().split())
        p -= 1  # Convert to 0-indexed
        q -= 1
        
        # Find which teams p and q belong to
        team_p = player_to_team[p]
        team_q = player_to_team[q]
        
        # Get team sizes
        size_p = len(teams[team_p])
        size_q = len(teams[team_q])
        
        # Calculate win probabilities
        # Team p wins with probability size_p / (size_p + size_q)
        # Team q wins with probability size_q / (size_p + size_q)
        
        total_size = size_p + size_q
        
        # Update expected wins for all players in team p
        for player in teams[team_p]:
            # Add size_p / total_size to expected wins
            expected_wins[player] = (expected_wins[player] + size_p * modinv(total_size)) % MOD
        
        # Update expected wins for all players in team q
        for player in teams[team_q]:
            # Add size_q / total_size to expected wins
            expected_wins[player] = (expected_wins[player] + size_q * modinv(total_size)) % MOD
        
        # Merge teams: move all players from team_q to team_p
        for player in teams[team_q]:
            teams[team_p].add(player)
            player_to_team[player] = team_p
        
        # Clear team_q
        teams[team_q].clear()
    
    # Print results
    print(' '.join(map(str, expected_wins)))

solve()