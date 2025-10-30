import itertools

def main():
    n = int(input())
    m_g = int(input())
    g_edges = set()
    for _ in range(m_g):
        u, v = map(int, input().split())
        if u > v:
            u, v = v, u
        g_edges.add((u, v))
    
    m_h = int(input())
    h_edges = set()
    for _ in range(m_h):
        a, b = map(int, input().split())
        if a > b:
            a, b = b, a
        h_edges.add((a, b))
    
    # Initialize A matrix (1-based indexing)
    A = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n):
        row = list(map(int, input().split()))
        idx = 0
        for j in range(i + 1, n + 1):
            A[i][j] = row[idx]
            A[j][i] = row[idx]
            idx += 1
    
    min_cost = float('inf')
    # Generate all permutations of vertices 1 to n
    for perm in itertools.permutations(range(1, n + 1)):
        target = set()
        for (u, v) in g_edges:
            a = perm[u - 1]
            b = perm[v - 1]
            if a > b:
                a, b = b, a
            target.add((a, b))
        # Calculate symmetric difference
        sym_diff = target.symmetric_difference(h_edges)
        cost = 0
        for (x, y) in sym_diff:
            cost += A[x][y]
        if cost < min_cost:
            min_cost = cost
    
    print(min_cost)

if __name__ == "__main__":
    main()