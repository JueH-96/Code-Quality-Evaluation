import itertools

def main():
    N = int(input())
    
    # Read graph G
    M_G = int(input())
    G_edges = set()
    for _ in range(M_G):
        u, v = map(int, input().split())
        if u > v:
            u, v = v, u
        G_edges.add((u, v))
    
    # Read graph H
    M_H = int(input())
    H_edges = set()
    for _ in range(M_H):
        a, b = map(int, input().split())
        if a > b:
            a, b = b, a
        H_edges.add((a, b))
    
    # Read the cost matrix
    cost = [[0] * (N + 1) for _ in range(N + 1)]  # 1-based indexing
    for i in range(1, N):
        parts = list(map(int, input().split()))
        j = i + 1
        for val in parts:
            cost[i][j] = val
            j += 1
    
    min_total = float('inf')
    
    # Generate all permutations of vertices
    for perm in itertools.permutations(range(1, N + 1)):
        target_edges = set()
        for u, v in G_edges:
            a = perm[u - 1]
            b = perm[v - 1]
            if a > b:
                a, b = b, a
            target_edges.add((a, b))
        
        # Compute symmetric difference
        diff = H_edges.symmetric_difference(target_edges)
        total = 0
        for a, b in diff:
            total += cost[a][b]
        
        if total < min_total:
            min_total = total
    
    print(min_total)

if __name__ == '__main__':
    main()