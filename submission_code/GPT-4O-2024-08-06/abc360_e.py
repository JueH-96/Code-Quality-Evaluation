def modinv(a, p):
    return pow(a, p - 2, p)

def solve():
    import sys
    input = sys.stdin.read
    MOD = 998244353
    
    data = input().strip()
    N, K = map(int, data.split())
    
    # Probability of black ball moving to a different position in one operation
    p_move = 2 * (N - 1) / (N * (N - 1))
    
    # Expected position after K operations
    expected_position = 1 + K * p_move
    
    # Express expected_position as a fraction P/Q
    P = 1 * (N * (N - 1)) + K * 2 * (N - 1)
    Q = N * (N - 1)
    
    # Reduce P/Q modulo MOD
    P %= MOD
    Q %= MOD
    
    # Find R such that R * Q ≡ P (mod MOD)
    R = (P * modinv(Q, MOD)) % MOD
    
    print(R)