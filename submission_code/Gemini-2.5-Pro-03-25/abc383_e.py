# YOUR CODE HERE
import sys

# Set higher recursion depth for potentially deep DSU find calls.
# Python's default recursion depth limit (often 1000) might be insufficient
# for graphs with up to N=2e5 vertices, even with path compression.
# Setting it to a larger value helps prevent RecursionError.
try:
    # A value like 2*N or slightly more should be safe. 
    # Example: For N=2e5, use 4e5 + 10.
    sys.setrecursionlimit(4 * 10**5 + 10) 
except Exception as e: 
    # Some systems might restrict setting a high recursion limit.
    # Pass silently if it fails. Path compression often keeps depth low anyway.
    pass

# DSU find operation with path compression
def find(parent, i):
    """
    Find the root of the set containing element i.
    Applies path compression heuristic for efficiency.
    parent: list where parent[i] is the parent of node i.
    i: The element whose set root is to be found.
    Returns the root of the set containing i.
    """
    if parent[i] == i:
        # If i is the root, return i
        return i
    # Recursively find the root and apply path compression:
    # set parent[i] directly to the root found.
    parent[i] = find(parent, parent[i])
    return parent[i]

# DSU union operation with union by rank and update component attributes
def union(parent, countA, countB, match_count, rank, i, j, w):
    """
    Union the sets containing elements i and j, using the union by rank heuristic.
    Updates component attributes: countA, countB (number of A/B vertices)
    and match_count (max possible matches within component).
    Calculates and returns the increase in the total minimum weight sum contributed
    by the matches enabled by this merge operation, weighted by the edge weight w.

    parent, countA, countB, match_count, rank: DSU state arrays.
    i, j: Elements (vertices) whose sets are to be merged.
    w: Weight of the edge causing this potential merge.
    Returns: Increase in total minimum weight sum due to this merge.
    """
    root_i = find(parent, i) # Find root of i's component
    root_j = find(parent, j) # Find root of j's component
    
    ans_increase = 0 # Initialize increase in sum to 0 for this potential merge
    
    # Only proceed if i and j are in different components
    if root_i != root_j: 
        
        # Retrieve current state (counts and match size) for both components
        ai, bi = countA[root_i], countB[root_i] # A/B counts for component i
        mi = match_count[root_i] # Max matches within component i
        
        aj, bj = countA[root_j], countB[root_j] # A/B counts for component j
        mj = match_count[root_j] # Max matches within component j

        # Calculate total A and B counts in the potential merged component
        a_new = ai + aj
        b_new = bi + bj
        
        # The maximum number of pairs (A_i, B_j) that can be matched within
        # the merged component is limited by the minimum of total A's and B's.
        new_matches_possible = min(a_new, b_new)
        
        # Calculate the number of *additional* matches made possible by merging.
        # This is the increase from the sum of matches possible in individual components 
        # (mi + mj) to the matches possible in the merged component (new_matches_possible).
        delta_matches = new_matches_possible - (mi + mj)
        
        # The cost associated with these newly enabled matches is w. This is because 
        # any path connecting vertices from the previously separate components requires 
        # traversing an edge of weight at least w. The current edge (u, v) with weight w
        # provides such a path, establishing w as the minimum possible maximum edge weight (f value)
        # for these newly connected pairs.
        ans_increase = delta_matches * w

        # Perform the union operation using the union by rank heuristic
        # This heuristic merges the tree with smaller rank into the tree with larger rank
        # to keep the tree depth low, improving efficiency of 'find'.
        if rank[root_i] < rank[root_j]:
             # Merge i's tree into j's tree. root_j becomes the new root.
             parent[root_i] = root_j
             # Update attributes for the new root (root_j)
             countA[root_j] = a_new
             countB[root_j] = b_new
             match_count[root_j] = new_matches_possible 
        elif rank[root_i] > rank[root_j]:
             # Merge j's tree into i's tree. root_i becomes the new root.
             parent[root_j] = root_i
             # Update attributes for the new root (root_i)
             countA[root_i] = a_new
             countB[root_i] = b_new
             match_count[root_i] = new_matches_possible
        else:
             # Ranks are equal. Arbitrarily merge j's tree into i's tree.
             # root_i becomes the new root. Increment its rank.
             parent[root_j] = root_i
             rank[root_i] += 1
             # Update attributes for the new root (root_i)
             countA[root_i] = a_new
             countB[root_i] = b_new
             match_count[root_i] = new_matches_possible
             
    return ans_increase # Return the calculated cost increase for this step

def solve():
    """Reads input, solves the problem using Kruskal-like algorithm with DSU, and prints the result."""
    N, M, K = map(int, sys.stdin.readline().split())
    
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        # Store edges as tuples (vertex1, vertex2, weight)
        edges.append((u, v, w))

    # Read the sequences A and B
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Sort edges by weight in non-decreasing order. This is crucial for the algorithm,
    # as it processes connections in order of increasing bottleneck capacity (max edge weight).
    edges.sort(key=lambda x: x[2])

    # Initialize DSU data structures. Use size N+1 for 1-based vertex indexing.
    parent = list(range(N + 1)) # parent[i] stores the parent of node i
    rank = [0] * (N + 1) # rank[i] stores the rank of the tree rooted at i (for union by rank)
    countA = [0] * (N + 1) # countA[i] stores the count of A_k vertices in the component rooted at i
    countB = [0] * (N + 1) # countB[i] stores the count of B_k vertices in the component rooted at i
    # match_count[i] stores the maximum number of pairs (A_k, B_l) that can be matched 
    # *within* the component rooted at i, considering only edges processed so far.
    match_count = [0] * (N + 1) 

    # Initialize vertex counts based on the input sequences A and B.
    # For each vertex x specified in A, increment its countA.
    for x in A:
        countA[x] += 1
    # For each vertex x specified in B, increment its countB.
    for x in B:
        countB[x] += 1
    
    # The initial match_count for each component (single vertex) is 0.
    # This is because the problem guarantees A_i != B_j for all i, j, meaning
    # the sets of vertices {A_k} and {B_k} are disjoint. Thus, a single vertex
    # cannot be both an A-type and a B-type, so min(countA[x], countB[x]) = 0 initially.

    total_min_weight_sum = 0 # Initialize the accumulator for the final answer

    # Process edges in increasing order of weight (Kruskal's algorithm style)
    for u, v, w in edges:
        # Attempt to union the components containing vertices u and v.
        # The 'union' function handles the logic: if they are already connected, it does nothing.
        # If they are not connected, it merges them, updates attributes, and returns the
        # cost contribution (delta_matches * w) of newly enabled matches.
        increase = union(parent, countA, countB, match_count, rank, u, v, w)
        
        # Add the calculated cost increase to the total sum.
        total_min_weight_sum += increase

    # After processing all edges, the total_min_weight_sum holds the minimum possible
    # sum of f(A_i, B_sigma(i)) over all permutations sigma of B.
    print(total_min_weight_sum)

# Execute the main solve function to run the program
solve()