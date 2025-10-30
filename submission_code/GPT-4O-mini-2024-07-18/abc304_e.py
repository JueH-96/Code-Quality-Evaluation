from sys import stdin, stdout
from collections import defaultdict, deque

def is_good_graph(N, edges, K, forbidden_pairs, Q, queries):
    # Build the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # Use BFS/DFS to find connected components
    visited = [False] * (N + 1)
    component = [-1] * (N + 1)
    comp_id = 0

    for node in range(1, N + 1):
        if not visited[node]:
            comp_id += 1
            queue = deque([node])
            visited[node] = True
            component[node] = comp_id
            
            while queue:
                current = queue.popleft()
                for neighbor in graph[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        component[neighbor] = comp_id
                        queue.append(neighbor)

    # Check forbidden pairs
    forbidden_set = set()
    for x, y in forbidden_pairs:
        if component[x] == component[y]:
            forbidden_set.add((min(x, y), max(x, y)))

    results = []
    for p, q in queries:
        if component[p] == component[q]:
            results.append("No")
        else:
            results.append("Yes")

    return results

def main():
    input = stdin.read
    data = input().splitlines()
    
    idx = 0
    N, M = map(int, data[idx].split())
    idx += 1
    
    edges = []
    for _ in range(M):
        u, v = map(int, data[idx].split())
        edges.append((u, v))
        idx += 1
    
    K = int(data[idx])
    idx += 1
    
    forbidden_pairs = []
    for _ in range(K):
        x, y = map(int, data[idx].split())
        forbidden_pairs.append((x, y))
        idx += 1
    
    Q = int(data[idx])
    idx += 1
    
    queries = []
    for _ in range(Q):
        p, q = map(int, data[idx].split())
        queries.append((p, q))
        idx += 1
    
    results = is_good_graph(N, edges, K, forbidden_pairs, Q, queries)
    
    stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()