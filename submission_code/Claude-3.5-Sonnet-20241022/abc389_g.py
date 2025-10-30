from collections import deque, defaultdict
from itertools import combinations

def count_distances_from_one(n, edges):
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from vertex 1
    distances = [-1] * (n + 1)
    distances[1] = 0
    queue = deque([1])
    
    while queue:
        v = queue.popleft()
        for u in adj[v]:
            if distances[u] == -1:
                distances[u] = distances[v] + 1
                queue.append(u)
    
    # Count vertices at even and odd distances
    even_count = sum(1 for d in distances[1:] if d >= 0 and d % 2 == 0)
    odd_count = sum(1 for d in distances[1:] if d >= 0 and d % 2 == 1)
    
    # Check if graph is connected and has equal counts
    return all(d >= 0 for d in distances[1:]) and even_count == odd_count

def solve(n, p):
    vertices = list(range(1, n + 1))
    all_edges = list(combinations(vertices, 2))
    max_edges = (n * (n - 1)) // 2
    result = []
    
    # For each possible number of edges
    for m in range(n - 1, max_edges + 1):
        count = 0
        # Generate all possible combinations of m edges
        for edges in combinations(all_edges, m):
            if count_distances_from_one(n, edges):
                count = (count + 1) % p
        result.append(count)
    
    return result

def main():
    # Read input
    n, p = map(int, input().split())
    
    # Solve and output result
    result = solve(n, p)
    print(*result)

if __name__ == "__main__":
    main()