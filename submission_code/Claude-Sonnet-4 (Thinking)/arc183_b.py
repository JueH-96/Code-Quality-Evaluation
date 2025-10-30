def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    # Union-Find to compute connected components
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union positions that are within distance K
    for i in range(n):
        for j in range(i + 1, min(n, i + k + 1)):
            union(i, j)
    
    # Group positions by connected component
    components = {}
    for i in range(n):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)
    
    # Check each component
    for positions in components.values():
        a_values = set(a[i] for i in positions)
        b_values = set(b[i] for i in positions)
        
        if not b_values.issubset(a_values):
            return "No"
    
    return "Yes"

t = int(input())
for _ in range(t):
    print(solve())