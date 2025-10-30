import itertools

MOD = 998244353

n, m = map(int, input().split())
b = list(map(int, input().split()))

# Find positions of -1
unknown_pos = []
for i in range(n):
    if b[i] == -1:
        unknown_pos.append(i)

q = len(unknown_pos)
total_sum = 0

def count_components(seq):
    # Union-Find implementation
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Add edges based on the condition A_i <= A_j for i < j
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] <= seq[j]:
                union(i, j)
    
    # Count connected components
    components = len(set(find(i) for i in range(n)))
    return components

# Generate all possible assignments
for assignment in itertools.product(range(1, m + 1), repeat=q):
    # Create the sequence B'
    b_prime = b[:]
    for i, val in enumerate(assignment):
        b_prime[unknown_pos[i]] = val
    
    # Calculate f(B')
    components = count_components(b_prime)
    total_sum = (total_sum + components) % MOD

print(total_sum)