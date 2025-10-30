MOD = 998244353

def modinv(x, p):
    # Fermat's little theorem: x^(p-1) = 1 (mod p) => x^(p-2) = x^(-1) (mod p)
    return pow(x, p - 2, p)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    matches = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N-1)]
    
    # Initialize the size of each player's team (initially each player is their own team)
    team_size = [1] * (N + 1)
    # Expected wins for each player
    expected_wins = [0] * (N + 1)
    
    # Process each match
    for p, q in matches:
        size_p = team_size[p]
        size_q = team_size[q]
        total_size = size_p + size_q
        
        # Calculate win probabilities
        prob_p_wins = size_p * modinv(total_size, MOD) % MOD
        prob_q_wins = size_q * modinv(total_size, MOD) % MOD
        
        # Update expected wins
        new_expected_wins_p = (expected_wins[p] + prob_p_wins) % MOD
        new_expected_wins_q = (expected_wins[q] + prob_q_wins) % MOD
        
        # Merge teams
        team_size[p] = team_size[q] = total_size
        expected_wins[p] = new_expected_wins_p
        expected_wins[q] = new_expected_wins_q
    
    # Print the expected wins for each player from 1 to N
    print(" ".join(str(expected_wins[i]) for i in range(1, N + 1)))

if __name__ == "__main__":
    main()