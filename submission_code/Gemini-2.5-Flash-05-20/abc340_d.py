import heapq
import sys

def solve():
    # Read N from standard input
    N = int(sys.stdin.readline())

    # Adjacency list to represent the graph.
    # adj[u] will store a list of (v, weight) tuples for edges from u to v.
    # We use 1-based indexing for stages, so the list size is N+1.
    adj = [[] for _ in range(N + 1)]

    # Read the N-1 lines of A_i, B_i, X_i
    # The loop runs for stage i from 1 to N-1
    for i in range(1, N):
        A_i, B_i, X_i = map(int, sys.stdin.readline().split())
        
        # Action 1: Spend A_i seconds to clear stage i, allows playing stage i+1.
        # Add an edge from i to i+1 with weight A_i.
        adj[i].append((i + 1, A_i))
        
        # Action 2: Spend B_i seconds to clear stage i, allows playing stage X_i.
        # Add an edge from i to X_i with weight B_i.
        adj[i].append((X_i, B_i))

    # Dijkstra's algorithm to find the shortest path from stage 1 to stage N

    # dist[k] will store the minimum time to be able to play stage k.
    # Initialize all distances to infinity.
    # We use float('inf') to represent infinity for distances.
    dist = [float('inf')] * (N + 1)
    
    # The cost to reach stage 1 is 0, as it's initially available.
    dist[1] = 0

    # Priority queue stores tuples of (current_cost, stage_id).
    # heapq implements a min-heap, so it always pops the element with the smallest cost.
    # Start with stage 1 and a cost of 0.
    pq = [(0, 1)] 

    while pq:
        # Pop the stage with the smallest known cost
        current_cost, u = heapq.heappop(pq)

        # If we have already found a shorter path to 'u',
        # this means the current popped element is a stale entry from a longer path.
        # So, we skip it.
        if current_cost > dist[u]:
            continue

        # Explore all neighbors 'v' of the current stage 'u'
        for v, weight in adj[u]:
            # Calculate the cost to reach 'v' through 'u'
            new_cost = dist[u] + weight
            
            # If this new path to 'v' is shorter than any previously known path to 'v'
            if new_cost < dist[v]:
                # Update the minimum cost to reach 'v'
                dist[v] = new_cost
                # Add 'v' to the priority queue with its new, smaller cost
                heapq.heappush(pq, (dist[v], v))
    
    # After Dijkstra's finishes, dist[N] will hold the minimum time
    # required to be able to play stage N.
    print(dist[N])

# Call the solve function to run the program
solve()