def solve():
    T = int(input())
    
    for _ in range(T):
        N, K = map(int, input().split())
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))
        
        # Union-Find with path compression
        parent = list(range(N))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Connect positions within distance K
        for i in range(N):
            for j in range(max(0, i-K), min(N, i+K+1)):
                union(i, j)
        
        # Group positions by their root
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(N):
            groups[find(i)].append(i)
        
        # Check each group
        possible = True
        for positions in groups.values():
            # Count values in A for this group
            count_A = defaultdict(int)
            for pos in positions:
                count_A[A[pos]] += 1
            
            # Count values in B for this group
            count_B = defaultdict(int)
            for pos in positions:
                count_B[B[pos]] += 1
            
            # Check if we have enough of each value
            for val, needed in count_B.items():
                if count_A[val] < needed:
                    possible = False
                    break
            
            if not possible:
                break
        
        print("Yes" if possible else "No")

solve()