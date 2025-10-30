def main():
    import sys
    input = sys.stdin.readline
    from itertools import combinations
    
    # Read input
    N, M, K = map(int, input().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, input().split())
        # Use 0-indexed vertices for internal representation.
        edges.append((u-1, v-1, w))
    
    # Function to check if the selected edges form a spanning tree
    # using Union-Find. Returns the sum of weights if valid, otherwise None.
    def is_spanning_tree(combo):
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            a = find(a)
            b = find(b)
            if a == b:
                return False
            parent[b] = a
            return True
        
        s = 0
        count = 0
        for idx in combo:
            u, v, w = edges[idx]
            if union(u, v):
                s += w
                count += 1
            else:
                # If union fails, it would create a cycle (shouldn't happen in MST candidate if we check connectivity after)
                pass
        # if we connected N-1 edges, it's a spanning tree.
        if count == N-1:
            return s
        else:
            return None

    ans = None
    # Enumerate all combinations of N-1 edges from M edges.
    for combo in combinations(range(M), N-1):
        total = is_spanning_tree(combo)
        if total is not None:
            cost = total % K
            if ans is None or cost < ans:
                ans = cost
                # early stopping if cost==0: cannot get lower than 0.
                if ans == 0:
                    break

    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()