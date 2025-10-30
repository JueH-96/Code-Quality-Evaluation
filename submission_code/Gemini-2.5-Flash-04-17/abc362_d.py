import sys
import heapq

# Read input
def read_int():
    return int(sys.stdin.readline())

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def solve():
    N, M = read_ints()
    A = read_ints() # Vertex weights, 0-indexed internally A[0]...A[N-1]

    # Adjacency list: adj[u] is a list of (v, weight) pairs
    # Vertices are 0-indexed internally (0 to N-1)
    adj = [[] for _ in range(N)]

    for _ in range(M):
        u, v, b = read_ints()
        # Adjust to 0-indexed
        u -= 1
        v -= 1
        
        # Add directed edges based on the problem's path weight definition
        # Moving from u to v via edge (u, v) with original weight B adds B + A[v] to the cumulative cost
        # The cost to transition from u to v is the edge weight plus the weight of the destination vertex v
        adj[u].append((v, b + A[v]))
        # Moving from v to u via edge (u, v) with original weight B adds B + A[u] to the cumulative cost
        adj[v].append((u, b + A[u]))

    # Dijkstra's algorithm
    # dist[i] will store the minimum cumulative cost from vertex 0 to vertex i
    # Initialize distances to infinity
    dist = [float('inf')] * N

    # Priority queue stores tuples (cost, vertex_index)
    # The priority queue automatically orders elements by the first item (cost)
    pq = []

    # The initial cumulative cost at the starting vertex (vertex 1, index 0) is just its weight A[0]
    start_node = 0 # Vertex 1 is index 0
    dist[start_node] = A[start_node]
    heapq.heappush(pq, (dist[start_node], start_node))

    while pq:
        current_cost, u = heapq.heappop(pq)

        # If the extracted cost is greater than the current minimum cost to u, skip
        # This means we've already found a shorter path to u and processed it
        if current_cost > dist[u]:
            continue

        # Explore neighbors
        for v, weight in adj[u]:
            # The cost to reach v via u is the cost to reach u + the weight of the edge transition from u to v
            new_cost = dist[u] + weight

            # If a shorter path to v is found
            if new_cost < dist[v]:
                dist[v] = new_cost
                heapq.heappush(pq, (new_cost, v))

    # The results for vertices 2 to N are dist[1] to dist[N-1]
    # We need to print these values
    results = dist[1:]

    # Print the results separated by spaces
    # Use print(*results) to unpack the list elements and print them with default space separator
    # Using map and join ensures consistent string output, especially useful for very large numbers
    print(*results)

# Run the solver function
if __name__ == "__main__":
    solve()