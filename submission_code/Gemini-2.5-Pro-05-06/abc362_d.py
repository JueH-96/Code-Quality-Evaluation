import heapq
import sys

# Use sys.stdin.readline for faster input
input = sys.stdin.readline

def solve():
    N, M = map(int, input().split())
    
    # Vertex weights. A[i] is the weight of 0-indexed vertex i.
    A = list(map(int, input().split()))

    # Adjacency list: adj[u] is a list of tuples (v, edge_weight_B_uv)
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v, b = map(int, input().split())
        # Adjust to 0-indexed vertices
        u -= 1
        v -= 1
        adj[u].append((v, b))
        adj[v].append((u, b)) # Graph is undirected

    # dist[i] stores the minimum path weight from vertex 0 to vertex i
    # Initialize all distances to infinity
    dist = [float('inf')] * N
    
    # The path from vertex 0 to vertex 0 is just vertex 0 itself.
    # Its weight is A[0] (weight of 0-indexed vertex 0).
    dist[0] = A[0]
    
    # Priority queue stores tuples: (current_path_weight, vertex_index)
    # heapq implements a min-heap.
    pq = [(A[0], 0)] 

    while pq:
        d, u = heapq.heappop(pq)

        # If a shorter path to u has already been found, skip this entry
        if d > dist[u]:
            continue

        # Explore neighbors of u
        for v, B_uv in adj[u]:
            # To extend the path from vertex 0 to u, then to v:
            # The current path (to u) has weight dist[u].
            # This dist[u] already includes weights of all vertices on the path to u (including u)
            # and all edges on that path.
            # To extend to v, we add the weight of edge (u,v) i.e., B_uv,
            # and the weight of vertex v i.e., A[v].
            new_dist_to_v = dist[u] + B_uv + A[v]
            
            if new_dist_to_v < dist[v]:
                dist[v] = new_dist_to_v
                heapq.heappush(pq, (dist[v], v))
    
    # The problem asks for path weights from original vertex 1 to original vertices 2, ..., N.
    # In our 0-indexed system, this is from vertex 0 to vertices 1, ..., N-1.
    # So we need to output dist[1], dist[2], ..., dist[N-1].
    
    # Prepare the list of results for printing
    # dist[0] is the path weight to vertex 1 (source).
    # dist[1] is the path weight to vertex 2.
    # ...
    # dist[N-1] is the path weight to vertex N.
    results_to_print = [str(dist[i]) for i in range(1, N)]
    
    # Print results separated by spaces
    print(" ".join(results_to_print))

if __name__ == '__main__':
    solve()