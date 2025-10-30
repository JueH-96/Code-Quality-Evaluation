from collections import defaultdict, deque
from itertools import combinations

def main():
    n = int(input())
    graph = defaultdict(list)
    edge_weights = {}
    total_weight = 0
    
    for _ in range(n-1):
        u, v, l = map(int, input().split())
        graph[u].append((v, l))
        graph[v].append((u, l))
        if u > v:
            u, v = v, u
        edge_weights[(u, v)] = l
        total_weight += l
    
    def find_path(start, end):
        if start == end:
            return [start]
        
        parent = [-1] * (n+1)
        visited = [False] * (n+1)
        queue = deque([start])
        visited[start] = True
        
        while queue:
            u = queue.popleft()
            if u == end:
                break
            for v, l in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
        
        path = []
        curr = end
        while curr != -1:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path
    
    def compute_steiner_tree_cost(vertices):
        steiner_edges = set()
        
        for v in vertices:
            if v == 1:
                continue
            path = find_path(1, v)
            for i in range(len(path) - 1):
                u, w = path[i], path[i+1]
                if u > w:
                    u, w = w, u
                steiner_edges.add((u, w))
        
        cost = 0
        for edge in steiner_edges:
            cost += edge_weights[edge]
        
        return cost
    
    for k in range(1, n+1):
        vertices = list(range(2, n+1))  # All vertices except 1
        max_cost = 0
        
        if k >= len(vertices):
            # Aoki chooses all vertices except 1
            max_cost = 2 * total_weight
        else:
            # Try all combinations of k vertices
            for subset in combinations(vertices, k):
                steiner_cost = compute_steiner_tree_cost(list(subset) + [1])
                cost = 2 * steiner_cost
                max_cost = max(max_cost, cost)
        
        print(max_cost)

if __name__ == "__main__":
    main()