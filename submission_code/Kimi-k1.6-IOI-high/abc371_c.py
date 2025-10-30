import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    
    n = int(data[idx])
    idx += 1
    
    m_g = int(data[idx])
    idx += 1
    
    g_edges = set()
    for _ in range(m_g):
        u = int(data[idx]) - 1
        v = int(data[idx + 1]) - 1
        idx += 2
        if u > v:
            u, v = v, u
        g_edges.add((u, v))
    
    m_h = int(data[idx])
    idx += 1
    
    h_edges = set()
    for _ in range(m_h):
        u = int(data[idx]) - 1
        v = int(data[idx + 1]) - 1
        idx += 2
        if u > v:
            u, v = v, u
        h_edges.add((u, v))
    
    A = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        for j in range(i + 1, n):
            A[i][j] = int(data[idx])
            idx += 1
    
    min_cost = float('inf')
    for perm in itertools.permutations(range(n)):
        cost = 0
        for i in range(n):
            for j in range(i + 1, n):
                a = perm[i]
                b = perm[j]
                if a > b:
                    g_has = (b, a) in g_edges
                else:
                    g_has = (a, b) in g_edges
                h_has = (i, j) in h_edges
                if g_has != h_has:
                    cost += A[i][j]
        if cost < min_cost:
            min_cost = cost
    
    print(min_cost)

if __name__ == "__main__":
    main()