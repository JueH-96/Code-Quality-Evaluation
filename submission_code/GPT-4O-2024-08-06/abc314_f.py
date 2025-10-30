def modinv(a, m):
    # Function to compute modular inverse of a under modulo m
    # Using Fermat's Little Theorem: a^(m-1) ≡ 1 (mod m) => a^(m-2) ≡ a^(-1) (mod m)
    return pow(a, m - 2, m)

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    matches = [(int(data[i * 2 + 1]), int(data[i * 2 + 2])) for i in range(N - 1)]
    
    # Initialize teams
    team_size = [1] * (N + 1)
    expected_wins = [0] * (N + 1)
    
    for p, q in matches:
        size_p = team_size[p]
        size_q = team_size[q]
        
        # Probability that team with player p wins
        prob_p_wins = size_p * modinv(size_p + size_q, MOD) % MOD
        # Probability that team with player q wins
        prob_q_wins = size_q * modinv(size_p + size_q, MOD) % MOD
        
        # Update expected wins for all players in team p
        for i in range(1, N + 1):
            if team_size[i] == size_p:
                expected_wins[i] = (expected_wins[i] + prob_p_wins) % MOD
        
        # Update expected wins for all players in team q
        for i in range(1, N + 1):
            if team_size[i] == size_q:
                expected_wins[i] = (expected_wins[i] + prob_q_wins) % MOD
        
        # Merge teams
        new_size = size_p + size_q
        for i in range(1, N + 1):
            if team_size[i] == size_p or team_size[i] == size_q:
                team_size[i] = new_size
    
    # Output the expected wins for each player
    print(' '.join(map(str, expected_wins[1:])))