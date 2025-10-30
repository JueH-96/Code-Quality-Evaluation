# YOUR CODE HERE
MOD = 998244353

def solve():
    N, Q = map(int, input().split())
    A = [0] + list(map(int, input().split()))
    
    # Process each query
    for _ in range(Q):
        u, v = map(int, input().split())
        
        # Calculate the sum of distances between u and v over all possible trees
        # Initialize with the contribution of the edges from u and v to their parents
        total_sum = 0
        
        # Add the weight of edge from u to its parent if u is not the root
        if u > 1:
            total_sum = (total_sum + A[u] * factorial(N-1)) % MOD
        
        # Add the weight of edge from v to its parent (always on the path)
        total_sum = (total_sum + A[v] * factorial(N-1)) % MOD
        
        # Add the contribution of edges from vertices between u and v
        for j in range(u + 1, v):
            # The probability that edge (j, parent(j)) is on the path is 1/j
            # So the expected contribution is A[j] / j
            contribution = (A[j] * factorial(N-1) * pow(j, MOD-2, MOD)) % MOD
            total_sum = (total_sum + contribution) % MOD
        
        print(total_sum)

def factorial(n):
    """Calculate factorial modulo MOD using precomputed values."""
    result = 1
    for i in range(2, n+1):
        result = (result * i) % MOD
    return result

# Entrypoint for the program
if __name__ == "__main__":
    solve()