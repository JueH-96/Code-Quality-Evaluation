def solve():
    N, M = map(int, input().split())
    B = list(map(int, input().split()))
    
    MOD = 998244353
    
    # Find positions with -1
    unknown_positions = []
    for i in range(N):
        if B[i] == -1:
            unknown_positions.append(i)
    
    q = len(unknown_positions)
    
    # If no -1s, just compute the answer directly
    if q == 0:
        result = count_components(B, N)
        print(result)
        return
    
    # Generate all possible assignments and sum components
    total = 0
    
    def generate_assignments(idx):
        nonlocal total
        if idx == q:
            # All -1s have been assigned, count components
            total = (total + count_components(B, N)) % MOD
            return
        
        pos = unknown_positions[idx]
        for val in range(1, M + 1):
            B[pos] = val
            generate_assignments(idx + 1)
        B[pos] = -1  # Reset for backtracking
    
    generate_assignments(0)
    print(total)

def count_components(B, N):
    # Use Union-Find to count connected components
    parent = list(range(N))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Add edges based on the condition
    for i in range(N):
        for j in range(i + 1, N):
            if B[i] <= B[j]:
                union(i, j)
    
    # Count unique components
    components = len(set(find(i) for i in range(N)))
    return components

solve()