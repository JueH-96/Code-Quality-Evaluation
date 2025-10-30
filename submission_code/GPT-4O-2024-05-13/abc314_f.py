MOD = 998244353

def modinv(a, p):
    return pow(a, p - 2, p)

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    matches = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N-1)]
    
    # Initialize the number of players in each team
    team_size = [1] * (N + 1)
    # Initialize the expected wins for each player
    expected_wins = [0] * (N + 1)
    
    for p, q in matches:
        a = team_size[p]
        b = team_size[q]
        
        # Calculate the expected wins for this match
        win_p = a * modinv(a + b, MOD) % MOD
        win_q = b * modinv(a + b, MOD) % MOD
        
        # Update the expected wins for each player in the teams
        expected_wins[p] = (expected_wins[p] + win_p) % MOD
        expected_wins[q] = (expected_wins[q] + win_q) % MOD
        
        # Merge the teams
        team_size[p] += team_size[q]
        team_size[q] = team_size[p]
    
    # Print the expected wins for each player
    print(" ".join(map(str, expected_wins[1:])))
    
solve()