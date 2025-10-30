MOD = 998244353

def mod_inv(a, mod):
    return pow(a, mod - 2, mod)

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    matches = []
    for i in range(N - 1):
        p = int(data[2 * i + 1])
        q = int(data[2 * i + 2])
        matches.append((p, q))
    
    # Initialize the expected wins for each player
    expected_wins = [0] * (N + 1)
    
    # Process each match in reverse order
    for p, q in reversed(matches):
        # Calculate the expected wins for the new team
        a = 1
        b = 1
        for i in range(1, N + 1):
            if i != p and i != q:
                a += expected_wins[i]
                b += expected_wins[i]
        
        # Update the expected wins for p and q
        expected_wins[p] = (a * (a + b) + b * (a + b)) // (2 * (a + b))
        expected_wins[q] = (b * (a + b) + a * (a + b)) // (2 * (a + b))
        
        # Remove the smaller team from the expected wins
        if a < b:
            expected_wins[p] = 0
        else:
            expected_wins[q] = 0
    
    # Print the expected wins for each player modulo 998244353
    for i in range(1, N + 1):
        print(expected_wins[i] % MOD, end=' ')

# Call the solve function
solve()