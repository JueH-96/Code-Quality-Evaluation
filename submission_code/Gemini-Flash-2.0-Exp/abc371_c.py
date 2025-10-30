def solve():
    n = int(input())
    mg = int(input())
    g_edges = []
    for _ in range(mg):
        u, v = map(int, input().split())
        g_edges.append((u - 1, v - 1))

    mh = int(input())
    h_edges = []
    for _ in range(mh):
        a, b = map(int, input().split())
        h_edges.append((a - 1, b - 1))

    a_costs = []
    for i in range(n - 1):
        a_costs.append(list(map(int, input().split())))

    def get_cost(i, j):
        if i > j:
            i, j = j, i
        return a_costs[i][j - i - 1]

    def are_isomorphic(perm):
        for i in range(n):
            for j in range(i + 1, n):
                g_edge = (i, j) in g_edges or (j, i) in g_edges
                h_edge = (perm[i], perm[j]) in h_edges or (perm[j], perm[i]) in h_edges
                if g_edge != h_edge:
                    return False
        return True

    import itertools
    
    min_cost = float('inf')

    for perm in itertools.permutations(range(n)):
        cost = 0
        temp_h_edges = set()
        for u, v in h_edges:
            temp_h_edges.add((u, v))
            temp_h_edges.add((v, u))

        
        target_edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in g_edges or (j, i) in g_edges:
                    target_edges.add((perm[i], perm[j]))
                    target_edges.add((perm[j], perm[i]))
        
        current_h_edges = set()
        for u, v in h_edges:
            current_h_edges.add((u,v))
            current_h_edges.add((v,u))

        
        for i in range(n):
            for j in range(i + 1, n):
                
                g_edge = (i, j) in g_edges or (j, i) in g_edges
                h_edge = (perm[i], perm[j]) in h_edges or (perm[j], perm[i]) in h_edges
                
                
                if g_edge:
                    if (perm[i], perm[j]) not in current_h_edges and (perm[j], perm[i]) not in current_h_edges:
                        pass
                    elif (perm[i], perm[j]) in current_h_edges or (perm[j], perm[i]) in current_h_edges:
                        cost += get_cost(perm[i], perm[j])
                else:
                    if (perm[i], perm[j]) not in current_h_edges and (perm[j], perm[i]) not in current_h_edges:
                        cost += get_cost(perm[i], perm[j])
                    elif (perm[i], perm[j]) in current_h_edges or (perm[j], perm[i]) in current_h_edges:
                        pass

        min_cost = min(min_cost, cost)

    print(min_cost)

solve()