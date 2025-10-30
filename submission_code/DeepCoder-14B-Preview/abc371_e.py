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
    h_current = set()
    for _ in range(m_h):
        a, b = map(int, input().split())
        if a > b:
            a, b = b, a
        h_current.add((a, b))
    
    # Read the A matrix
    a = [[0] * (n + 1) for _ in range(n + 1)]  # a[i][j] for i < j
    current_i = 1
    for _ in range(n - 1):
        line = list(map(int, input().split()))
        current_j = current_i + 1
        for val in line:
            a[current_i][current_j] = val
            current_j += 1
        current_i += 1
    
    min_cost = float('inf')
    
    # Generate all possible permutations of the vertices
    for perm in itertools.permutations(range(1, n + 1)):
        desired = set()
        for u, v in g_edges:
            a_p = perm[u - 1]
            b_p = perm[v - 1]
            if a_p > b_p:
                a_p, b_p = b_p, a_p
            desired.add((a_p, b_p))
        
        cost = 0
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                in_h = (i, j) in h_current
                in_desired = (i, j) in desired
                if in_h != in_desired:
                    cost += a[i][j]
        
        if cost < min_cost:
            min_cost = cost
    
    print(min_cost)

if __name__ == '__main__':
    main()