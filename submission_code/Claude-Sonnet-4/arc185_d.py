MOD = 998244353

def modinv(a):
    return pow(a, MOD-2, MOD)

def solve():
    N, M = map(int, input().split())
    
    # For this specific tree structure, we can derive the formula
    # The expected cover time equals the sum of hitting times from 0 to each vertex
    
    result = 0
    
    # Process each layer
    for layer in range(M):
        layer_start = layer * N + 1
        layer_end = min((layer + 1) * N, N * M)
        
        if layer_start > N * M:
            break
            
        layer_size = layer_end - layer_start + 1
        
        # Expected hitting time to vertices in this layer
        # This follows from the electrical network theory
        if layer == 0:
            # Direct children of root
            hitting_time = 2 * modinv(N)
        else:
            # For layer k, hitting time is 2*(2k+1)/N
            hitting_time = (2 * (2 * layer + 1) * modinv(N)) % MOD
        
        result = (result + layer_size * hitting_time) % MOD
    
    return result

N, M = map(int, input().split())
print(solve())