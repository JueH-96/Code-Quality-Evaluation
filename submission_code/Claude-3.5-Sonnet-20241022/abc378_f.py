from collections import defaultdict, deque

def get_degree_map(N, edges):
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return degree

def find_cycle(N, edges, new_edge):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    graph[new_edge[0]].append(new_edge[1])
    graph[new_edge[1]].append(new_edge[0])
    
    visited = set()
    parent = {}
    cycle = []
    
    def dfs(node, prev):
        visited.add(node)
        parent[node] = prev
        
        for next_node in graph[node]:
            if next_node == prev:
                continue
            if next_node in visited:
                curr = node
                cycle.append(next_node)
                while curr != next_node:
                    cycle.append(curr)
                    curr = parent[curr]
                return True
            if dfs(next_node, node):
                return True
        return False
    
    dfs(1, -1)
    return cycle

def solve():
    # Read input
    N = int(input())
    edges = []
    for _ in range(N-1):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    # Get initial degree of each vertex
    degree = get_degree_map(N, edges)
    
    # Try all possible new edges
    answer = 0
    seen = set()
    for u in range(1, N+1):
        for v in range(u+1, N+1):
            # Skip if edge already exists
            if any((u == a and v == b) or (u == b and v == a) for a, b in edges):
                continue
            
            # Add new edge and find cycle
            cycle = find_cycle(N, edges, (u, v))
            
            # Check if all vertices in cycle have degree 3
            new_degree = degree.copy()
            new_degree[u] += 1
            new_degree[v] += 1
            
            if all(new_degree[vertex] == 3 for vertex in cycle):
                answer += 1
    
    print(answer)

solve()