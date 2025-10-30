from collections import defaultdict, deque
import itertools

def find_shortest_path(graph, start, end):
    dist = {start: 0}
    queue = deque([start])
    prev = {start: None}
    
    while queue:
        u = queue.popleft()
        for v, edges in graph[u].items():
            if v not in dist:
                dist[v] = dist[u] + min(t for _, t in edges)
                prev[v] = u
                queue.append(v)
                
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        curr = prev[curr]
    return path[::-1]

def solve_query(N, graph, required_edges):
    # Try all possible permutations of required edges
    min_total_time = float('inf')
    
    for perm in itertools.permutations(required_edges):
        # For each edge in permutation, try both directions
        for directions in itertools.product([0, 1], repeat=len(perm)):
            total_time = 0
            curr_pos = 1  # Start at island 1
            
            # Follow the path using required edges
            for edge, direction in zip(perm, directions):
                u, v, t = edge
                if direction:
                    u, v = v, u
                
                if curr_pos != u:
                    # Find shortest path to u
                    path = find_shortest_path(graph, curr_pos, u)
                    # Add time for shortest path
                    for i in range(len(path)-1):
                        total_time += min(t for _, t in graph[path[i]][path[i+1]])
                
                # Cross the required edge
                total_time += t
                curr_pos = v
            
            # Find path from last position to N if needed
            if curr_pos != N:
                path = find_shortest_path(graph, curr_pos, N)
                for i in range(len(path)-1):
                    total_time += min(t for _, t in graph[path[i]][path[i+1]])
            
            min_total_time = min(min_total_time, total_time)
    
    return min_total_time

# Read input
N, M = map(int, input().split())

# Create graph
graph = defaultdict(lambda: defaultdict(list))
edges = []
for i in range(M):
    u, v, t = map(int, input().split())
    graph[u][v].append((i+1, t))
    graph[v][u].append((i+1, t))
    edges.append((u, v, t))

Q = int(input())
for _ in range(Q):
    K = int(input())
    bridge_nums = list(map(int, input().split()))
    required_edges = [edges[i-1] for i in bridge_nums]
    print(solve_query(N, graph, required_edges))