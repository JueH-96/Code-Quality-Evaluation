# -*- coding: utf-8 -*-
import sys
# networkx might not be preinstalled depending on the competition environment.
# If running locally, you might need to install networkx using pip: pip install networkx
import networkx as nx 

# Function to solve a single test case
def solve():
    # Read N (number of cakes) and K (number of pairs) from standard input
    # Use sys.stdin.readline for faster input compared to input()
    line = sys.stdin.readline().split()
    N = int(line[0])
    K = int(line[1])
    
    # Read cake attributes (X, Y, Z) for each of the N cakes
    # Store the attributes as a list of tuples for easy access
    cakes = []
    for _ in range(N):
        line = sys.stdin.readline().split()
        cakes.append((int(line[0]), int(line[1]), int(line[2]))) # Store as tuples (X, Y, Z)

    # Initialize the maximum total price found so far across all strategies
    max_total_price = 0

    # Define the 8 sign combinations for the linear scoring function.
    # These combinations represent directions in the 3D attribute space
    # and are used to rank cakes in 8 different ways.
    # The combinations are (sX, sY, sZ) where each element is either 1 or -1.
    # Example: (1, 1, 1) corresponds to scoring by X + Y + Z, 
    #          (1, 1, -1) corresponds to scoring by X + Y - Z, etc.
    signs = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
             (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]

    # Iterate through each of the 8 sign combinations
    for sx, sy, sz in signs:
        # Calculate a score for each cake using the current linear combination (sx*X + sy*Y + sz*Z)
        # This score is used to rank the cakes
        scored_cakes = []
        for i in range(N):
            score = sx * cakes[i][0] + sy * cakes[i][1] + sz * cakes[i][2]
            # Store a tuple of (calculated_score, original_cake_index)
            scored_cakes.append((score, i))

        # Sort the cakes in descending order based on the calculated score.
        # This identifies the top cakes according to the current scoring criterion.
        # Sorting takes O(N log N) time.
        scored_cakes.sort(key=lambda item: item[0], reverse=True)

        # Select the top 2K cakes from the sorted list based on their original indices.
        # These 2K cakes are the candidates for forming the K pairs in this iteration.
        chosen_indices = [item[1] for item in scored_cakes[:2 * K]]

        # Create a graph data structure to represent the chosen cakes and potential pairs.
        # We will use this graph to find a maximum weight perfect matching among the chosen cakes.
        G = nx.Graph()
        # Add the original indices of the chosen cakes as nodes in the graph.
        # Using original indices helps map back to cake attributes later.
        G.add_nodes_from(chosen_indices)

        # Add edges between all distinct pairs of chosen cakes.
        # This forms a complete graph on the 2K chosen vertices.
        # The weight of an edge between cake u (with original index u_idx) and cake v (with original index v_idx)
        # is defined as the price of pairing them: max(Xu+Xv, Yu+Yv, Zu+Zv).
        # This part builds the complete graph on 2K vertices. It takes O((2K)^2) time.
        # Storing the graph explicitly also requires O((2K)^2) memory for the edges.
        # Given sum of N constraints, O((2K)^2) memory might be a concern for very large N in a single test case.
        num_chosen = 2 * K
        for i in range(num_chosen):
            for j in range(i + 1, num_chosen):
                u_idx = chosen_indices[i] # Original index of the i-th cake in the chosen_indices list
                v_idx = chosen_indices[j] # Original index of the j-th cake in the chosen_indices list
                
                # Get the attribute values for the two cakes using their original indices from the 'cakes' list
                cake_u = cakes[u_idx]
                cake_v = cakes[v_idx]

                # Calculate the price (weight) of pairing these two cakes
                weight = max(cake_u[0] + cake_v[0],
                             cake_u[1] + cake_v[1],
                             cake_u[2] + cake_v[2])
                
                # Add an edge between node u_idx and node v_idx with the calculated weight.
                # In networkx, edge weights are stored in a dictionary associated with the edge.
                G.add_edge(u_idx, v_idx, weight=weight)

        # Find the Maximum Weight Perfect Matching in the created graph.
        # networkx.max_weight_matching implements the Blossom algorithm, which is a standard algorithm
        # for finding maximum weight matchings in general graphs.
        # For a complete graph with 2K vertices, a perfect matching always exists.
        # We are maximizing the sum of edge weights, which is standard for max_weight_matching with positive weights.
        # The edge weights calculated (price) are guaranteed to be non-negative since X, Y, Z >= 0.
        # 'with_matching=True' is needed to get the actual set of edges (pairs of nodes) that form the matching.
        # The time complexity of this step on a dense graph with V=2K vertices is roughly O(V^3) or O(V^2 log V) in the worst case.
        # This is another potentially expensive step.
        matching = nx.max_weight_matching(G, with_matching=True)

        # Calculate the total price of the found matching by summing the weights of the matched edges.
        current_total_price = 0
        for u, v in matching:
             # Access the weight of the matched edge (u, v) from the graph G
             current_total_price += G[u][v]['weight']

        # Update the overall maximum total price found so far across all 8 sign combinations.
        max_total_price = max(max_total_price, current_total_price)

    # Print the final maximum total price for this test case.
    # Python 3's built-in integer type handles arbitrarily large values,
    # which is necessary as the total price can exceed 32-bit or 64-bit integer limits depending on K and attribute values.
    print(max_total_price)

# Read the number of test cases from standard input
T = int(sys.stdin.readline())

# Process each test case by calling the solve() function
for _ in range(T):
    solve()