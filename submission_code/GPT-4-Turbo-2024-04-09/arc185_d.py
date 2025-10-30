MOD = 998244353

def modinv(x, mod):
    """ Compute modular inverse of x under modulo `mod` """
    return pow(x, mod - 2, mod)

def solve():
    import sys
    input = sys.stdin.read
    N, M = map(int, input().split())
    
    # Total vertices
    total_vertices = N * M + 1
    
    # Expected number of moves to paint all vertices
    # We use the formula derived for the expected number of moves:
    # E = 2 * (N * M) + 1
    # This is because each vertex from 1 to N*M has an expected 2 visits (1 to paint, 1 to return)
    # and vertex 0 is already painted.
    
    # Result is 2 * (N * M) + 1
    result = (2 * N * M + 1) % MOD
    
    print(result)