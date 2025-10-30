MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    # The expected number of operations is N * M + (N * M) * (N * M - 1) / 2
    # Because each vertex is painted once, and each edge is traversed once in expectation
    # But the actual formula is more complex due to the tree structure
    
    # The expected number of operations is the sum of the expected number of steps to paint each vertex
    # For a tree, the expected number of steps to paint all vertices is the sum of the expected number of steps to paint each vertex, which is the sum of the reciprocals of the degrees of the vertices
    
    # However, in this specific tree, the degrees are as follows:
    # Vertex 0 has degree N
    # Each vertex i (1 <= i <= N*M) has degree 2 if it is not a leaf, or 1 if it is a leaf
    
    # The expected number of operations is the sum of the expected number of steps to paint each vertex, which is the sum of the reciprocals of the degrees of the vertices
    
    # For vertex 0, the expected number of steps to paint it is 1 / N
    # For each vertex i (1 <= i <= N*M), the expected number of steps to paint it is 1 / (degree of i)
    
    # The degree of vertex i is 2 if it is not a leaf, otherwise 1
    
    # The leaves are the vertices i where i > N*(M-1)
    
    # So, the expected number of operations is:
    # 1 / N + sum_{i=1}^{N*M} 1 / (degree of i)
    
    # The degree of vertex i is 2 if i <= N*(M-1), otherwise 1
    
    # So, the sum is:
    # 1 / N + sum_{i=1}^{N*(M-1)} 1 / 2 + sum_{i=N*(M-1)+1}^{N*M} 1 / 1
    
    # Which simplifies to:
    # 1 / N + (N*(M-1)) / 2 + (N*M - N*(M-1)) / 1
    
    # Which further simplifies to:
    # 1 / N + (N*(M-1)) / 2 + N
    
    # So, the expected number of operations is:
    # 1 / N + (N*(M-1)) / 2 + N
    
    # To compute this modulo 998244353, we need to compute the modular inverses
    
    inv_N = pow(N, MOD-2, MOD)
    inv_2 = pow(2, MOD-2, MOD)
    
    term1 = inv_N
    term2 = (N * (M - 1)) % MOD * inv_2 % MOD
    term3 = N % MOD
    
    total = (term1 + term2 + term3) % MOD
    
    print(total)

if __name__ == "__main__":
    main()