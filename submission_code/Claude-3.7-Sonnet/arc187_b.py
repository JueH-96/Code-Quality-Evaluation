import itertools

def solve():
    N, M = map(int, input().split())
    B = list(map(int, input().split()))
    
    # Find positions of -1 in the sequence
    minus_one_positions = [i for i, val in enumerate(B) if val == -1]
    q = len(minus_one_positions)
    
    total_sum = 0
    MOD = 998244353
    
    def count_components(B_prime):
        # Initialize Union-Find data structure
        parent = list(range(N))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        # Add edges between vertices i and j if B_prime[i] <= B_prime[j]
        for i in range(N):
            for j in range(i+1, N):
                if B_prime[i] <= B_prime[j]:
                    a_root = find(i)
                    b_root = find(j)
                    if a_root != b_root:
                        parent[a_root] = b_root
        
        # Count connected components
        return sum(1 for i in range(N) if parent[i] == i)
    
    # If there are no -1 values in B, just count components for B itself
    if q == 0:
        return count_components(B)
    
    # Generate all possible combinations of replacements for -1
    values = list(range(1, M+1))
    for vals in itertools.product(values, repeat=q):
        B_prime = B.copy()
        for i, pos in enumerate(minus_one_positions):
            B_prime[pos] = vals[i]
        
        total_sum = (total_sum + count_components(B_prime)) % MOD
    
    return total_sum

print(solve())