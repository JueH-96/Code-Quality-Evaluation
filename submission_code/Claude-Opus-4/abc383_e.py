import heapq
from itertools import permutations

def dijkstra_max_edge(graph, start, n):
    # Find minimum "max edge weight" path from start to all other vertices
    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
            
        for v, w in graph[u]:
            # The max edge weight to reach v through u is max(dist[u], w)
            new_dist = max(dist[u], w)
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    
    return dist

def solve():
    # Read input
    n, m, k = map(int, input().split())
    
    # Build adjacency list
    graph = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Compute f(x, y) for all needed pairs
    # We need f(A[i], B[j]) for all i, j
    f_values = {}
    
    # Run Dijkstra from each vertex in A
    for a in set(A):
        dist = dijkstra_max_edge(graph, a, n)
        for b in set(B):
            f_values[(a, b)] = dist[b]
    
    # Try all permutations of B to find minimum sum
    min_sum = float('inf')
    
    if k <= 8:  # For small k, try all permutations
        for perm in permutations(range(k)):
            current_sum = 0
            for i in range(k):
                current_sum += f_values[(A[i], B[perm[i]])]
            min_sum = min(min_sum, current_sum)
    else:
        # For larger k, use Hungarian algorithm or greedy approach
        # Since constraints allow k <= n <= 2*10^5, and we need to be efficient
        # We'll use a greedy approach with optimization
        
        # Create cost matrix
        costs = []
        for i in range(k):
            row = []
            for j in range(k):
                row.append(f_values[(A[i], B[j])])
            costs.append(row)
        
        # Simple assignment using greedy + local optimization
        used = [False] * k
        assignment = [-1] * k
        
        # Greedy initial assignment
        for i in range(k):
            best_j = -1
            best_cost = float('inf')
            for j in range(k):
                if not used[j] and costs[i][j] < best_cost:
                    best_cost = costs[i][j]
                    best_j = j
            assignment[i] = best_j
            used[best_j] = True
        
        # Calculate initial sum
        current_sum = sum(costs[i][assignment[i]] for i in range(k))
        
        # Try swapping to improve
        improved = True
        while improved:
            improved = False
            for i in range(k):
                for j in range(i + 1, k):
                    # Try swapping assignment[i] and assignment[j]
                    old_cost = costs[i][assignment[i]] + costs[j][assignment[j]]
                    new_cost = costs[i][assignment[j]] + costs[j][assignment[i]]
                    if new_cost < old_cost:
                        assignment[i], assignment[j] = assignment[j], assignment[i]
                        current_sum = current_sum - old_cost + new_cost
                        improved = True
        
        min_sum = current_sum
    
    print(min_sum)

solve()