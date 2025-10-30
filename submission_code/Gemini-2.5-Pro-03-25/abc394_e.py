# YOUR CODE HERE
import collections
import sys

# Main function to solve the problem
def solve():
    # Read graph size N from standard input
    N = int(sys.stdin.readline())
    # Read the N x N character matrix describing edges and labels
    C = [sys.stdin.readline().strip() for _ in range(N)]

    # Represent the graph using adjacency lists.
    # adj[i] contains list of pairs (j, label) for edges i -> j
    adj = [[] for _ in range(N)]
    # Use a reverse adjacency list to efficiently find predecessors of a vertex.
    # rev_adj[j] contains list of pairs (i, label) for edges i -> j
    rev_adj = [[] for _ in range(N)]

    # Populate adjacency lists based on input matrix C
    # Use 0-based indexing for vertices internally (0 to N-1)
    for i in range(N):
        for j in range(N):
            # Check if an edge exists from i to j (character is not '-')
            if C[i][j] != '-':
                # Add edge to forward adjacency list for vertex i
                adj[i].append((j, C[i][j]))
                # Add edge to reverse adjacency list for vertex j (stores incoming edge from i)
                rev_adj[j].append((i, C[i][j]))

    # Initialize distance matrix `dist` with infinity. Using float('inf') is standard.
    # `dist[i][j]` will store the length of the shortest palindromic path from vertex i to vertex j.
    INF = float('inf')
    dist = [[INF] * N for _ in range(N)]
    
    # Initialize a deque (double-ended queue) for the Breadth-First Search (BFS).
    # The queue will store pairs of vertices (u, v). Each pair represents the endpoints
    # of a palindromic path `u -> ... -> v` whose length `dist[u][v]` has been determined.
    q = collections.deque()

    # Base case initialization for paths of length 0.
    # The path from any vertex i to itself has length 0. The empty path has an empty string label, which is a palindrome.
    # This is always the shortest possible path from i to i.
    for i in range(N):
        dist[i][i] = 0
        # Add the state (i, i) representing the length 0 path to the queue.
        q.append((i, i))

    # Base case initialization for paths of length 1.
    # A path i -> j using a single edge has length 1. Its label is a single character.
    # Single character strings are always palindromes.
    for i in range(N):
        # Iterate through all outgoing edges from vertex i
        for neighbor_j, label in adj[i]:
             # We only need to consider paths of length 1 between distinct vertices (i != neighbor_j).
             # If i == neighbor_j, the length 0 path `dist[i][i] = 0` is already found and is shorter.
             if i != neighbor_j:
                 # If the path of length 1 is shorter than the current recorded distance for (i, neighbor_j)
                 # (This check is initially true since dist[i][neighbor_j] starts at INF)
                 if dist[i][neighbor_j] > 1: 
                     dist[i][neighbor_j] = 1
                     # Add this state (i, neighbor_j) to the queue. Finding this path might enable
                     # finding shorter paths of length 3, 5, ... later.
                     q.append((i, neighbor_j))

    # Perform BFS to find shortest palindromic paths of length >= 2.
    # The algorithm works by iteratively extending known palindromic paths.
    # If we have a palindromic path `u -> ... -> v` of length `k`, we can try to extend it to
    # `u_prev -> u -> ... -> v -> v_next` of length `k+2`. This new path is palindromic
    # if the labels of the added edges `(u_prev, u)` and `(v, v_next)` are the same.
    while q:
        # Get the next state (pair of endpoints `u`, `v`) from the front of the queue
        u, v = q.popleft()
        # Retrieve the length `k` of the shortest palindromic path found so far from `u` to `v`
        current_k = dist[u][v] 

        # Try extending the path `u -> ... -> v` by adding an edge at the beginning and an edge at the end.
        # The structure becomes: `u_prev -> u ... v -> v_next`.
        
        # Iterate through all predecessors `u_prev` of vertex `u` using the reverse adjacency list.
        # `label1` is the label of the edge `(u_prev, u)`.
        for u_prev, label1 in rev_adj[u]:
            # Iterate through all successors `v_next` of vertex `v` using the forward adjacency list.
            # `label2` is the label of the edge `(v, v_next)`.
            for v_next, label2 in adj[v]:
                # Check if the labels of the two extending edges match.
                if label1 == label2:
                    # If labels match, the extended path `u_prev -> ... -> v_next` is also palindromic.
                    # Its length is `current_k + 2`.
                    new_dist = current_k + 2
                    
                    # Check if this newly found path from `u_prev` to `v_next` is shorter than any
                    # previously recorded path length for this pair.
                    if new_dist < dist[u_prev][v_next]:
                        # If it's shorter, update the shortest distance found so far.
                        dist[u_prev][v_next] = new_dist
                        # Add the updated state (endpoints `u_prev`, `v_next`) to the queue
                        # because finding this shorter path might lead to discovering other shorter paths later.
                        q.append((u_prev, v_next))

    # After the BFS completes, the `dist` matrix contains the lengths of the shortest palindromic paths.
    # If `dist[i][j]` is still `INF`, it means there is no palindromic path from vertex `i` to `j`.

    # Prepare the final answer matrix for output according to the problem specification.
    # Replace `INF` values with -1 to indicate that no palindromic path exists.
    final_ans = [[-1] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if dist[i][j] != INF:
                # Ensure the output values are integers.
                final_ans[i][j] = int(dist[i][j]) 

    # Print the final answer matrix row by row, with elements separated by spaces.
    for i in range(N):
        # The `*` operator unpacks the list `final_ans[i]` into separate arguments for print,
        # which results in space-separated output by default.
        print(*(final_ans[i]))

# Call the main function to start the process when the script is executed.
solve()