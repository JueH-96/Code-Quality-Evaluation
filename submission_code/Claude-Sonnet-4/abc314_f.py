# YOUR CODE HERE
def modinv(a, m=998244353):
    return pow(a, m-2, m)

def solve():
    n = int(input())
    matches = []
    for _ in range(n-1):
        p, q = map(int, input().split())
        matches.append((p-1, q-1))  # Convert to 0-indexed
    
    # Initially each player is in their own team
    # team_of[i] = which team player i belongs to
    team_of = list(range(n))
    # team_members[i] = set of players in team i
    team_members = [set([i]) for i in range(n)]
    # team_size[i] = size of team i
    team_size = [1] * n
    
    # expected_wins[i] = expected number of wins for player i
    expected_wins = [0] * n
    
    for match_idx, (p, q) in enumerate(matches):
        # Find which teams p and q belong to
        team_p = team_of[p]
        team_q = team_of[q]
        
        size_p = team_size[team_p]
        size_q = team_size[team_q]
        total_size = size_p + size_q
        
        # Probability team_p wins
        prob_p_wins = size_p * modinv(total_size) % 998244353
        # Probability team_q wins  
        prob_q_wins = size_q * modinv(total_size) % 998244353
        
        # Add expected wins for all players in team_p
        for player in team_members[team_p]:
            expected_wins[player] = (expected_wins[player] + prob_p_wins) % 998244353
            
        # Add expected wins for all players in team_q
        for player in team_members[team_q]:
            expected_wins[player] = (expected_wins[player] + prob_q_wins) % 998244353
        
        # Merge teams - team_q merges into team_p
        for player in team_members[team_q]:
            team_of[player] = team_p
            team_members[team_p].add(player)
        
        team_size[team_p] += team_size[team_q]
        team_members[team_q].clear()
        team_size[team_q] = 0
    
    print(*expected_wins)

solve()