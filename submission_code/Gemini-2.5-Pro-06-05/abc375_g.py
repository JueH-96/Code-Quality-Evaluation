import sys
import heapq

# YOUR CODE HERE
def solve():
    """
    Solves the AtCoder Shortest Distance problem.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read problem size
    try:
        N, M = map(int, input().split())
    except (IOError, ValueError):
        # Exit if input is empty
        return

    # Adjacency list for the graph
    adj = [[] for _ in range(N)]
    # List to store original edges for final iteration
    edges = []

    # Read edges and build the graph
    for _ in range(M):
        u, v, c = map(int, input().split())
        u -= 1
        v -= 1
        adj[u].append((v, c))
        adj[v].append((u, c))
        edges.append((u, v, c))

    # Use two large prime moduli to avoid hash collisions for path counting
    MOD1 = 10**9 + 7
    MOD2 = 998244353
    MODS = [MOD1, MOD2]

    # A large number to represent infinity
    inf = float('inf')

    def dijkstra_with_counts(start_node):
        """
        Runs Dijkstra's algorithm from a start node to find shortest distances
        and the number of shortest paths to all other nodes.
        Path counts are calculated modulo several primes.
        """
        # dist[i]: shortest distance from start_node to node i
        dist = [inf] * N
        # counts[m][i]: number of shortest paths from start_node to i, modulo MODS[m]
        counts = [[0] * N for _ in range(len(MODS))]
        
        dist[start_node] = 0
        for i in range(len(MODS)):
            counts[i][start_node] = 1
        
        # Priority queue for Dijkstra: stores (distance, node)
        pq = [(0, start_node)]
        
        while pq:
            d, u = heapq.heappop(pq)
            
            # If we found a shorter path already, skip
            if d > dist[u]:
                continue
            
            # Explore neighbors
            for v_neighbor, w in adj[u]:
                if dist[u] + w < dist[v_neighbor]:
                    # Found a new shorter path
                    dist[v_neighbor] = dist[u] + w
                    for i in range(len(MODS)):
                        counts[i][v_neighbor] = counts[i][u]
                    heapq.heappush(pq, (dist[v_neighbor], v_neighbor))
                elif dist[u] + w == dist[v_neighbor]:
                    # Found another path of the same shortest length
                    for i in range(len(MODS)):
                        counts[i][v_neighbor] = (counts[i][v_neighbor] + counts[i][u]) % MODS[i]
        
        return dist, counts

    # Run Dijkstra from the start city (1, index 0)
    dist1, counts1 = dijkstra_with_counts(0)
    # Run Dijkstra from the end city (N, index N-1)
    distN, countsN = dijkstra_with_counts(N - 1)
    
    # The shortest path length from city 1 to city N
    shortest_path_len = dist1[N - 1]
    
    # Total number of shortest paths from 1 to N, modulo the primes
    total_counts = [c[N - 1] for c in counts1]

    # The problem guarantees N is reachable, so shortest_path_len is finite.
    
    # For each road, determine if it lies on EVERY shortest path
    for u_edge, v_edge, c_edge in edges:
        is_critical = False
        
        # A road (u, v) is on a shortest path if dist(1,u) + len(u,v) + dist(v,N) = dist(1,N).
        # We check both possible directions of traversal along the path.
        # These two cases are mutually exclusive for edges with c > 0.
        
        # Case 1: The path traverses the edge as u -> v
        if dist1[u_edge] + c_edge + distN[v_edge] == shortest_path_len:
            # This edge is on at least one shortest path.
            # Check if it's on ALL shortest paths.
            # This happens if num_sp(1->u) * num_sp(v->N) == total_num_sp(1->N).
            match = True
            for i in range(len(MODS)):
                paths_thru = (counts1[i][u_edge] * countsN[i][v_edge]) % MODS[i]
                if paths_thru != total_counts[i]:
                    match = False
                    break
            if match:
                is_critical = True

        # Case 2: The path traverses the edge as v -> u
        elif dist1[v_edge] + c_edge + distN[u_edge] == shortest_path_len:
            # Similar check for the other direction.
            match = True
            for i in range(len(MODS)):
                paths_thru = (counts1[i][v_edge] * countsN[i][u_edge]) % MODS[i]
                if paths_thru != total_counts[i]:
                    match = False
                    break
            if match:
                is_critical = True
                
        # If the edge is critical, removing it will change the shortest path.
        if is_critical:
            print("Yes")
        else:
            print("No")

solve()