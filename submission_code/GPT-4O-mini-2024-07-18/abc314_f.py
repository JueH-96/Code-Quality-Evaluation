def modinv(a, p):
    return pow(a, p - 2, p)

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    matches = [tuple(map(int, line.split())) for line in data[1:N]]
    
    MOD = 998244353
    
    # Initialize the sizes and expected wins
    size = [1] * (N + 1)
    expected_wins = [0] * (N + 1)
    
    for p, q in matches:
        a = size[p]
        b = size[q]
        
        # Calculate probabilities
        prob_p_wins = a * modinv(a + b, MOD) % MOD
        prob_q_wins = b * modinv(a + b, MOD) % MOD
        
        # Update expected wins
        expected_wins[p] = (expected_wins[p] + prob_p_wins) % MOD
        expected_wins[q] = (expected_wins[q] + prob_q_wins) % MOD
        
        # Merge teams
        size[p] += size[q]
        size[q] = 0  # Mark q's team as merged
    
    # Output the expected wins for each player
    print(' '.join(str(expected_wins[i]) for i in range(1, N + 1)))

if __name__ == "__main__":
    main()