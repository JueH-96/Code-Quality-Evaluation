def modular_inverse(a, m):
    """Calculate the modular inverse of a mod m using Fermat's Little Theorem."""
    return pow(a, m - 2, m)

def solve(n, matches):
    MOD = 998244353
    
    # Initialize data structures
    parent = list(range(n + 1))  # Union-find parent array
    team_size = [1] * (n + 1)  # Size of each team
    
    def find(x):
        """Find the representative of the team containing player x."""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    # Record all match outcomes to replay later
    match_outcomes = []
    
    # First pass: simulate matches and record outcomes
    for pi, qi in matches:
        team_pi = find(pi)
        team_qi = find(qi)
        
        # Record match data
        match_outcomes.append((team_pi, team_qi, team_size[team_pi], team_size[team_qi]))
        
        # Merge teams
        parent[team_qi] = team_pi
        team_size[team_pi] += team_size[team_qi]
        team_size[team_qi] = 0
    
    # Reset for second pass
    parent = list(range(n + 1))
    team_size = [1] * (n + 1)
    expected_wins = [0] * (n + 1)
    
    # Second pass: calculate expected wins
    for team_pi, team_qi, size_pi, size_qi in match_outcomes:
        total_size = size_pi + size_qi
        
        # Calculate win probabilities (modular arithmetic)
        prob_pi_wins = (size_pi * modular_inverse(total_size, MOD)) % MOD
        prob_qi_wins = (size_qi * modular_inverse(total_size, MOD)) % MOD
        
        # Find all players in each team
        players_in_pi = []
        players_in_qi = []
        
        for i in range(1, n + 1):
            team_i = find(i)
            if team_i == team_pi:
                players_in_pi.append(i)
            elif team_i == team_qi:
                players_in_qi.append(i)
        
        # Update expected wins for all players in each team
        for player in players_in_pi:
            expected_wins[player] = (expected_wins[player] + prob_pi_wins) % MOD
            
        for player in players_in_qi:
            expected_wins[player] = (expected_wins[player] + prob_qi_wins) % MOD
        
        # Merge teams
        parent[team_qi] = team_pi
        team_size[team_pi] += team_size[team_qi]
        team_size[team_qi] = 0
    
    return expected_wins[1:]

def main():
    n = int(input())
    matches = []
    for _ in range(n - 1):
        pi, qi = map(int, input().split())
        matches.append((pi, qi))
    
    result = solve(n, matches)
    print(*result)

if __name__ == "__main__":
    main()