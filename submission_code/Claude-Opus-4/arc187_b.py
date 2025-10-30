def find_parent(parent, i):
    if parent[i] != i:
        parent[i] = find_parent(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    px = find_parent(parent, x)
    py = find_parent(parent, y)
    
    if px == py:
        return
    
    if rank[px] < rank[py]:
        parent[px] = py
    elif rank[px] > rank[py]:
        parent[py] = px
    else:
        parent[py] = px
        rank[px] += 1

def count_components(n, edges):
    parent = list(range(n))
    rank = [0] * n
    
    for u, v in edges:
        union(parent, rank, u, v)
    
    components = len(set(find_parent(parent, i) for i in range(n)))
    return components

def solve():
    MOD = 998244353
    
    n, m = map(int, input().split())
    b = list(map(int, input().split()))
    
    # Find positions of -1s
    unknown_pos = []
    for i in range(n):
        if b[i] == -1:
            unknown_pos.append(i)
    
    q = len(unknown_pos)
    
    if q == 0:
        # No -1s, just compute f(B) directly
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if b[i] <= b[j]:
                    edges.append((i, j))
        print(count_components(n, edges))
        return
    
    # DP approach
    # dp[mask] = sum of f(B') for all assignments to the first k -1s (represented by mask)
    total = 0
    
    # Generate all possible assignments
    for assignment in range(m ** q):
        # Convert assignment to actual values
        values = []
        temp = assignment
        for _ in range(q):
            values.append((temp % m) + 1)
            temp //= m
        
        # Create the sequence B'
        b_prime = b[:]
        for i, pos in enumerate(unknown_pos):
            b_prime[pos] = values[i]
        
        # Build edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if b_prime[i] <= b_prime[j]:
                    edges.append((i, j))
        
        # Count components
        components = count_components(n, edges)
        total = (total + components) % MOD
    
    print(total)

solve()