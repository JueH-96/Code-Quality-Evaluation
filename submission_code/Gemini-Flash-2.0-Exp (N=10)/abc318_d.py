def solve():
    n = int(input())
    weights = []
    for i in range(n - 1):
        weights.append(list(map(int, input().split())))
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if i < j:
                if i == 0:
                    edges.append((i, j, weights[0][j-1]))
                else:
                    edges.append((i, j, weights[i][j-i-1]))
    
    edges.sort(key=lambda x: x[2], reverse=True)
    
    max_weight = 0
    for i in range(1 << len(edges)):
        current_weight = 0
        chosen_edges = []
        nodes_used = set()
        
        for j in range(len(edges)):
            if (i >> j) & 1:
                u, v, w = edges[j]
                if u not in nodes_used and v not in nodes_used:
                    current_weight += w
                    chosen_edges.append((u,v))
                    nodes_used.add(u)
                    nodes_used.add(v)
                else:
                    current_weight = -1
                    break
        if current_weight > max_weight:
            max_weight = current_weight
    print(max_weight)

solve()