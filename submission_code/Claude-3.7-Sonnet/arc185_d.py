def solve():
    N, M = map(int, input().split())
    MOD = 998244353
    
    # Calculate the expected operations to reach a vertex at depth d
    def expected_operations(d):
        return d * (N + d - 1)
    
    # Calculate total expected operations
    total = 0
    for d in range(1, M+1):
        # Number of vertices at depth d
        vertices_at_depth = N
        # Expected operations for each vertex at depth d
        ops_per_vertex = expected_operations(d)
        # Add to total
        total += vertices_at_depth * ops_per_vertex
    
    # Calculate modular inverse of 1 using Fermat's little theorem
    # We need to compute (total / 1) % MOD which is just (total * 1^(MOD-2)) % MOD
    result = total % MOD
    
    return result

if __name__ == "__main__":
    print(solve())