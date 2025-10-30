# YOUR CODE HERE
import sys

# Function to read input faster from standard input
def fast_readline():
    # Reads a line from stdin and strips leading/trailing whitespace including newline
    return sys.stdin.readline().strip()

def fast_read_int_list():
    # Reads a line, splits it by whitespace, converts parts to integers, and returns as a list
    return list(map(int, fast_readline().split()))

def solve():
    # Read N (number of vertices) and M (number of edges) from the first line of input
    N, M = fast_read_int_list()

    # Initialize adjacency list for all vertices from 1 to N.
    # We use a dictionary where keys are vertex indices (1 to N) and values are lists of adjacent vertices.
    # This handles potentially disconnected vertices correctly.
    adj = {i: [] for i in range(1, N + 1)}
    
    # Read M edges from input and populate the adjacency list.
    for _ in range(M):
        u, v = fast_read_int_list()
        # Since the graph is undirected, add edge in both directions.
        adj[u].append(v)
        adj[v].append(u) 

    # Read weights W for each vertex. Pad with a 0 at index 0 to align with 1-based vertex indexing.
    W = [0] + fast_read_int_list() 
    # Read initial piece counts A for each vertex. Pad with a 0 at index 0.
    A = [0] + fast_read_int_list() 

    # Precomputation Step: Compute the set S_x for each vertex x.
    # S_x is the set of neighbors y of x that receive a piece when an operation is performed on x.
    # To maximize the total number of operations, we choose S_x to maximize its size |S_x| 
    # subject to the constraint sum(W[y] for y in S_x) < W[x].
    # This maximum size set is obtained by greedily selecting neighbors with the smallest weights.
    S_set = {} # Dictionary to store the computed set S_x for each vertex x. Using a set allows efficient lookups.

    for x in range(1, N + 1):
        neighbors = adj[x] # Get the list of neighbors for vertex x
        
        # If vertex x has no neighbors, its set S_x is empty.
        if not neighbors:
             S_set[x] = set()
             continue

        # Create a list of pairs (neighbor_weight, neighbor_id) for sorting purposes.
        neighbor_weights = []
        for y in neighbors:
            neighbor_weights.append((W[y], y))
        
        # Sort the neighbors based on their weights in ascending order.
        neighbor_weights.sort()

        current_weight_sum = 0
        # Use a set for `chosen_neighbors` to represent S_x. Sets provide O(1) average time complexity for membership checks later.
        chosen_neighbors = set() 
        
        # Iterate through the sorted neighbors and add them to S_x greedily.
        for weight_y, y in neighbor_weights:
            # Check if adding this neighbor keeps the total weight sum strictly less than W[x].
            if current_weight_sum + weight_y < W[x]:
                current_weight_sum += weight_y
                chosen_neighbors.add(y)
            else:
                # Since neighbors are sorted by weight, any subsequent neighbor 
                # has weight >= weight_y. Adding it would also result in the sum being >= W[x]. 
                # Therefore, we cannot add any more neighbors and can stop the process here.
                break 
        
        # Store the computed set S_x for vertex x.
        S_set[x] = chosen_neighbors

    # Computation Step: Calculate the total number of operations ('count') for each vertex.
    # The total number of operations performed at vertex v is the total number of pieces processed at v.
    
    # Get vertex indices (from 1 to N).
    indices = list(range(1, N + 1))
    
    # Sort the vertex indices based on their weights W[i] in descending order.
    # Processing vertices in this order ensures that when we compute count[v], 
    # the counts for all neighbors x that could potentially send a piece to v 
    # (which requires W[x] > W[v]) have already been computed. This works because pieces only flow from higher weight vertices to lower weight vertices.
    sorted_indices = sorted(indices, key=lambda i: W[i], reverse=True)

    # Initialize count array to store the total number of operations performed AT each vertex.
    # Python's integers have arbitrary precision, preventing overflow issues even for large counts.
    count = [0] * (N + 1) 

    # Process vertices one by one in the order determined by descending weights.
    for v in sorted_indices:
        # Calculate the total number of pieces received by vertex v from operations performed on its neighbors.
        # A piece is received by v from neighbor x if and only if v belongs to the set S_x.
        current_sum_from_neighbors = 0
        
        neighbors_of_v = adj[v] # Get neighbors of the current vertex v.
        for x in neighbors_of_v:
            # Check if vertex v is included in the precomputed set S_x associated with neighbor x.
            # This check is efficient (O(1) average time) because S_set[x] is a hash set.
            if v in S_set[x]:
                 # If v is in S_x, it means an operation at vertex x generates a piece that lands on vertex v.
                 # The total number of such pieces generated over the entire process is equal to the total number of operations performed at x, which is count[x].
                 # We add count[x] to the sum of pieces received by v.
                 # Note: The condition W[x] > W[v] must hold for v to be in S_x (since W[v] is positive).
                 # Because we process vertices in descending order of weight, count[x] is guaranteed
                 # to have been computed before we compute count[v].
                 current_sum_from_neighbors += count[x]

        # The total number of operations performed at vertex v (count[v]) equals
        # its initial number of pieces A[v] plus the total number of pieces it receives from all neighbors over the entire process.
        count[v] = A[v] + current_sum_from_neighbors

    # The final answer is the total maximum number of operations possible across the entire graph.
    # This is the sum of operations performed at all vertices.
    total_ops = sum(count[1:]) # Sum counts for vertices 1 through N. index 0 is unused padding.
    
    # Print the final answer to standard output.
    print(total_ops)

# Execute the main solve function to run the program.
solve()