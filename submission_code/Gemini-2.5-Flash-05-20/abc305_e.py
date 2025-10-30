import sys
import heapq

def solve():
    N, M, K = map(int, sys.stdin.readline().split())

    # Adjacency list for the graph. Vertices are 1-indexed.
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Store guard information (position p_i, stamina h_i)
    guards = []
    for _ in range(K):
        p, h = map(int, sys.stdin.readline().split())
        guards.append((p, h))

    # max_stamina_at_vertex[v] will store the maximum stamina a guard can have
    # upon reaching vertex v. Initialized to -1 to indicate not yet reached
    # or not guarded effectively.
    # Vertices are 1-indexed, so we use N+1 size and ignore index 0.
    max_stamina_at_vertex = [-1] * (N + 1)

    # Priority queue stores tuples: (-stamina_remaining, vertex)
    # We use negative stamina because heapq is a min-heap, and we want to
    # prioritize vertices with higher remaining stamina (i.e., smaller negative value).
    pq = []

    # Initialize priority queue with guards' starting positions and their stamina.
    # Since p_i are distinct, each p will be encountered only once as a starting guard location.
    for p, h in guards:
        # A guard is at p with stamina h. This is the initial "effective stamina" at p.
        max_stamina_at_vertex[p] = h
        heapq.heappush(pq, (-h, p))

    # Dijkstra-like algorithm to propagate maximum effective stamina
    while pq:
        # Pop the vertex with the highest current_stamina (smallest negative value)
        neg_current_stamina, u = heapq.heappop(pq)
        current_stamina = -neg_current_stamina

        # If we have already found a path to 'u' that leaves more stamina,
        # then this path is suboptimal, so skip. This handles redundant entries
        # in the PQ and ensures we process the best path first.
        if current_stamina < max_stamina_at_vertex[u]:
            continue

        # Explore neighbors of u
        for v in adj[u]:
            # Calculate the stamina remaining if we move from u to v.
            # Each edge traversal reduces stamina by 1.
            next_stamina = current_stamina - 1

            # If reaching 'v' via 'u' results in more remaining stamina than
            # any path found previously for 'v', update and push to PQ.
            if next_stamina > max_stamina_at_vertex[v]:
                max_stamina_at_vertex[v] = next_stamina
                heapq.heappush(pq, (-next_stamina, v))

    # Collect all guarded vertices.
    # A vertex 'v' is guarded if max_stamina_at_vertex[v] is non-negative.
    guarded_vertices = []
    for i in range(1, N + 1):
        if max_stamina_at_vertex[i] >= 0:
            guarded_vertices.append(i)

    # Print the output in the specified format
    sys.stdout.write(str(len(guarded_vertices)) + '
')
    sys.stdout.write(' '.join(map(str, guarded_vertices)) + '
')

# Call the solve function to run the program
solve()