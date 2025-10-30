import sys
import collections

# Increase recursion depth for DFS if needed, but BFS is used here which is iterative.
# sys.setrecursionlimit(2000000)

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Bipartite graph nodes:
    # Nodes 0 to N-1 represent sets S_1 to S_N (1-indexed input sets)
    # Nodes N to N+M-1 represent elements 1 to M (1-indexed input elements)
    num_nodes = N + M
    adj = [[] for _ in range(num_nodes)]

    # Build the adjacency list for the bipartite graph
    # An edge exists between a set node and an element node if the element is in the set.
    for i in range(N):
        A = int(sys.stdin.readline())
        elements = list(map(int, sys.stdin.readline().split()))
        set_node = i # Node index for S_{i+1} (0 to N-1)
        for element in elements:
            # Element j (1-indexed) maps to node N + j - 1
            element_node = N + element - 1 # Node index for element
            
            # Add edges between the set node and each element node it contains.
            # Since the graph is undirected, add edge in both directions.
            adj[set_node].append(element_node)
            adj[element_node].append(set_node)

    # Define start and target nodes for BFS
    # Element 1 is node N + 1 - 1 = N
    # Element M is node N + M - 1
    start_node = N # Node representing element 1
    target_node = N + M - 1 # Node representing element M

    # Initialize distance array for BFS. dist[u] will store the shortest distance
    # (number of edges) from the start_node to node u. Initialize with -1 (unvisited).
    dist = [-1] * num_nodes
    
    # Initialize BFS queue
    q = collections.deque()

    # Start BFS from the node representing element 1
    dist[start_node] = 0
    q.append(start_node)

    # Run BFS
    while q:
        u = q.popleft()

        # If we reached the target node (element M), we found the shortest path
        if u == target_node:
            break # Exit BFS loop

        # Explore neighbors of the current node u
        for v in adj[u]:
            # If neighbor v has not been visited yet (distance is -1)
            if dist[v] == -1:
                dist[v] = dist[u] + 1 # Set distance: one more edge than dist[u]
                q.append(v) # Add neighbor to the queue to visit its neighbors later

    # After BFS, check the distance to the target node (element M)
    if dist[target_node] == -1:
        # If the target node is unreachable from the start node (distance is still -1),
        # it's impossible to obtain a set containing both 1 and M.
        print(-1)
    else:
        # The shortest path length between element 1 node and element M node is L_min
        L_min = dist[target_node]

        # The path in the bipartite graph alternates between element nodes and set nodes.
        # A path from an element node (1) to another element node (M) must have an even length.
        # The structure is E -> S -> E -> S -> ... -> S -> E.
        # For a path of length L_min, the number of nodes is L_min + 1.
        # The nodes at even distances (0, 2, 4, ...) are element nodes.
        # The nodes at odd distances (1, 3, 5, ...) are set nodes.
        # The number of set nodes on a shortest path of length L_min is L_min / 2.
        # To merge K sets into one requires K-1 operations.
        # The minimum number of operations required is (Number of set nodes on shortest path) - 1.
        # Minimum operations = (L_min / 2) - 1.
        # Since L_min is the distance between two nodes of the same type (element), L_min must be even.
        # Using integer division // 2 is correct.

        min_operations = (L_min // 2) - 1
        print(min_operations)

solve()