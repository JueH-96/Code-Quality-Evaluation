def solve():
    n = int(input())
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    def get_leaves(graph):
        leaves = []
        for i in range(1, len(graph)):
            if len(graph[i]) <= 1:
                leaves.append(i)
        return leaves

    def remove_leaf(graph, leaf):
        neighbors = graph[leaf]
        for neighbor in neighbors:
            graph[neighbor].remove(leaf)
        graph[leaf] = []

    q = [(adj, 0)]
    visited = {tuple(tuple(sorted(x)) for x in adj)}
    
    while q:
        curr_adj, steps = q.pop(0)
        
        if len(curr_adj[1]) == 0:
            print(steps)
            return
        
        leaves = get_leaves(curr_adj)
        
        for leaf in leaves:
            new_adj = [list(x) for x in curr_adj]
            remove_leaf(new_adj, leaf)
            
            adj_tuple = tuple(tuple(sorted(x)) for x in new_adj)
            if adj_tuple not in visited:
                q.append((new_adj, steps + 1))
                visited.add(adj_tuple)

solve()