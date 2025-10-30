def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))

    def is_alkane(graph):
        if not graph:
            return False
        
        degrees = {}
        for u, v in graph:
            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1
        
        degree_values = list(degrees.values())
        if not degree_values:
            return False

        if all(d in [1, 4] for d in degree_values) and any(d == 4 for d in degree_values):
            
            num_vertices = len(set([u for edge in graph for u in edge]))
            
            adj = {node: [] for node in set([u for edge in graph for u in edge])}
            for u, v in graph:
                adj[u].append(v)
                adj[v].append(u)
            
            visited = set()
            q = [list(adj.keys())[0]]
            visited.add(q[0])
            
            count = 0
            while q:
                u = q.pop(0)
                count += 1
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            
            return count == num_vertices
        else:
            return False

    def find_all_subgraphs(edges, vertices):
        subgraphs = []
        for i in range(1 << len(edges)):
            subgraph_edges = []
            subgraph_vertices = set()
            for j in range(len(edges)):
                if (i >> j) & 1:
                    subgraph_edges.append(edges[j])
                    subgraph_vertices.add(edges[j][0])
                    subgraph_vertices.add(edges[j][1])
            
            if all(v in vertices for v in subgraph_vertices):
                subgraphs.append(subgraph_edges)
        return subgraphs
    
    max_vertices = -1
    
    for r in range(1, n + 1):
        for comb in combinations(range(1, n + 1), r):
            vertices = set(comb)
            
            possible_edges = []
            for u, v in edges:
                if u in vertices and v in vertices:
                    possible_edges.append((u, v))
            
            subgraphs = find_all_subgraphs(possible_edges, vertices)
            
            for subgraph in subgraphs:
                if is_alkane(subgraph):
                    num_vertices = len(set([u for edge in subgraph for u in edge]))
                    max_vertices = max(max_vertices, num_vertices)

    print(max_vertices)

from itertools import combinations

solve()