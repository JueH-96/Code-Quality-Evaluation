import collections

def solve():
    # Read N (number of nodes) and M (number of pairs/edges)
    N, M = map(int, input().split())

    # Read sequences A and B
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Step 1: Handle impossible case where A_i == B_i
    # If X_{A_i} must be different from X_{B_i}, and A_i == B_i,
    # then X_{A_i} must be different from X_{A_i}, which is impossible.
    for i in range(M):
        if A[i] == B[i]:
            print("No")
            return

    # Step 2: Build the graph using an adjacency list
    # The graph nodes are 1-indexed from 1 to N.
    # An edge (u, v) means X_u and X_v must have different values.
    adj = collections.defaultdict(list)
    for i in range(M):
        # Add undirected edge between A[i] and B[i]
        adj[A[i]].append(B[i])
        adj[B[i]].append(A[i])

    # Step 3: Perform Bipartite Check using BFS
    # colors array:
    # -1: Uncolored
    # 0: First color
    # 1: Second color
    colors = [-1] * (N + 1)
    is_bipartite = True

    # Iterate through all nodes to handle disconnected components
    for start_node in range(1, N + 1):
        # If the node hasn't been colored yet, start a BFS from it
        if colors[start_node] == -1:
            q = collections.deque()
            q.append(start_node)
            colors[start_node] = 0  # Assign the first color (0) to the starting node

            while q:
                u = q.popleft()  # Get the current node from the front of the queue

                # Explore all neighbors of the current node u
                for v in adj[u]:
                    if colors[v] == -1:  # If neighbor v is uncolored
                        colors[v] = 1 - colors[u]  # Assign the opposite color
                        q.append(v)  # Add v to the queue for further exploration
                    elif colors[v] == colors[u]:  # If neighbor v has the same color as u
                        # This indicates an odd-length cycle (e.g., u-v edge where u and v have same color)
                        # The graph is not bipartite.
                        is_bipartite = False
                        break  # Break from the inner loop (exploring neighbors)
                
                if not is_bipartite:
                    break  # Break from the current BFS (queue is still not empty, but result is determined)
        
        if not is_bipartite:
            break  # Break from the outer loop (iterating through all possible start_nodes)

    # Step 4: Print the result
    if is_bipartite:
        print("Yes")
    else:
        print("No")

# Call the solve function to run the program
solve()