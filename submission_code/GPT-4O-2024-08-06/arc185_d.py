def solve():
    import sys
    input = sys.stdin.read
    data = input().strip()
    N, M = map(int, data.split())
    
    MOD = 998244353
    
    # Calculate the expected number of operations
    total_vertices = N * M + 1
    expected_operations = 0
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Calculate the expected number of operations
    for i in range(1, total_vertices):
        # The number of children for node i is min(N, total_vertices - i)
        children_count = min(N, total_vertices - i)
        expected_operations += inv[children_count]
        expected_operations %= MOD
    
    # Multiply by N to account for the expected number of operations
    expected_operations *= N
    expected_operations %= MOD
    
    # Output the result
    print(expected_operations)