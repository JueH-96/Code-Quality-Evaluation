# YOUR CODE HERE
import sys
from bisect import bisect_right

# Setting a slightly higher recursion depth limit just in case.
# The actual required depth is logarithmic in M (size of coordinate space), 
# which is at most N+Q. For N+Q = 4*10^5, log2(4*10^5) is approx 18.6. 
# Python's default limit (often 1000) is usually sufficient.
# Uncomment the line below if stack overflow occurs, although it's unlikely for this problem structure.
# sys.setrecursionlimit(400010) 


class Node:
    """Node for the persistent segment tree."""
    # Using __slots__ can slightly reduce memory usage per node object, which might be helpful for large N.
    __slots__ = ['value', 'left', 'right']

    def __init__(self, value=0):
        """Initializes a Node with a default value of 0."""
        # self.value stores the maximum LIS length ending with a value within this node's range
        self.value = value  
        self.left = None # Reference to the left child node
        self.right = None # Reference to the right child node

# Global variables for coordinate compression related data.
# Using globals simplifies function signatures in a competitive programming context.
coords = [] # List to store sorted unique values
val_to_rank = {} # Dictionary mapping value to its rank (0-based index)
M = 0 # Size of the coordinate compressed space (number of unique values)

def build(l, r):
    """Builds an empty segment tree for range [l, r]. Returns the root node."""
    node = Node() # Create a new node, default value is 0
    if l == r: # Base case: leaf node
        return node
    # Recursively build left and right subtrees
    mid = (l + r) // 2
    node.left = build(l, mid)
    node.right = build(mid + 1, r)
    # The value of an internal node in an empty tree is also 0
    return node

def update(prev_node, l, r, pos, new_val):
    """
    Updates the value at position 'pos' to 'new_val' in the segment tree.
    This function implements persistence: it returns the root of the *new* version 
    of the tree, created by copying the path from the root to the updated leaf 
    and reusing unchanged subtrees from the previous version (`prev_node`).
    """
    
    # Create a new node for the current version. Its value will be computed based on children.
    node = Node()
    
    if l == r:
        # Base case: We have reached the leaf node corresponding to 'pos'.
        # Update the value. Direct assignment is used as explained in thought process.
        node.value = new_val
        return node

    # Determine the middle point to decide direction (left/right)
    mid = (l + r) // 2
    
    # Initialize children references by copying from the previous version's node.
    # This is the core mechanism of path-copying persistence.
    node.left = prev_node.left
    node.right = prev_node.right

    # Recursively call update on the appropriate child branch.
    # The call returns a *new* child node reference, which we assign.
    if pos <= mid:
        # Update path goes left
        node.left = update(prev_node.left, l, mid, pos, new_val)
    else:
        # Update path goes right
        node.right = update(prev_node.right, mid + 1, r, pos, new_val)
    
    # After updating a child, recalculate this node's value based on its children.
    # The value should be the maximum of its children's values.
    # Add safe checks for non-existent children (shouldn't happen in standard build).
    left_val = node.left.value if node.left else 0
    right_val = node.right.value if node.right else 0
    node.value = max(left_val, right_val)
    
    # Return the newly created/updated node as the root of this subtree for this version.
    return node

def query(node, l, r, query_l, query_r):
    """
    Queries the maximum value in the segment tree within the range [query_l, query_r].
    Operates on the specific tree version rooted at 'node'.
    """
    
    # Handle base cases and invalid ranges:
    # 1. node is None (should not happen with proper build/update)
    # 2. Query range [query_l, query_r] does not overlap with node's range [l, r]
    # 3. Invalid query range where query_l > query_r
    if node is None or query_l > r or query_r < l or query_l > query_r:
        # Return 0, the identity element for max operation, indicating no contribution from this path/range.
        return 0 
    
    # If node's range [l, r] is fully contained within the query range [query_l, query_r]
    if query_l <= l and r <= query_r:
        # Return the precomputed maximum value stored in this node.
        return node.value

    # Node's range partially overlaps or contains the query range.
    # Need to query children.
    mid = (l + r) // 2
    
    # Recursively query left and right children and return the maximum of their results.
    left_max = query(node.left, l, mid, query_l, query_r)
    right_max = query(node.right, mid + 1, r, query_l, query_r)
    
    return max(left_max, right_max)


def solve():
    # Read N and Q
    N, Q = map(int, sys.stdin.readline().split())
    # Read the sequence A
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read all queries and store them along with X_i values for coordinate compression
    queries_data = []
    query_X_values = [] # Store X_i values from queries
    for i in range(Q):
        R_i, X_i = map(int, sys.stdin.readline().split())
        queries_data.append((R_i, X_i))
        query_X_values.append(X_i)

    # --- Coordinate Compression ---
    global coords, val_to_rank, M # Declare usage of global variables
    
    # Collect all unique values from A and all query X_i values.
    # Using a set avoids duplicates efficiently.
    all_vals = set(A)
    all_vals.update(query_X_values) # Add all X_i values to the set
    
    # Create a sorted list of unique values. This forms our coordinate space.
    coords = sorted(list(all_vals)) 
    # Create a dictionary mapping each unique value to its rank (0-based index).
    val_to_rank = {val: i for i, val in enumerate(coords)} 
    # M is the total number of unique values, defining the size of our segment tree range.
    M = len(coords) 

    # Handle edge case where M=0 (e.g., N=0). Unlikely given constraints 1 <= N, Q.
    if M == 0:
       # If there are no elements, any LIS length is 0.
       for _ in range(Q):
           print(0) 
       return

    # --- Persistent Segment Tree Construction ---
    
    # Build the initial persistent segment tree (version 0) representing the state before processing any elements.
    # It covers the coordinate compressed range [0, M-1]. All values initially 0.
    # `roots` list stores the root node of the segment tree after processing each prefix A[0...i-1].
    # roots[0] corresponds to the empty prefix.
    roots = [build(0, M - 1)] 

    # Process sequence A element by element (from index 0 to N-1).
    # For each element A[j], create a new version of the segment tree.
    for j in range(N):
        A_j = A[j] # Current element value from sequence A
        idx_j = val_to_rank[A_j] # Rank (0-based index) of the current element
        
        # --- LIS Logic ---
        # Query the *latest* tree version (roots[-1], representing state after A[0...j-1])
        # to find the maximum LIS length ending with a value *strictly less* than A_j.
        # The query range for ranks corresponding to values < A_j is [0, idx_j - 1].
        k = 0 # Initialize max LIS length found ending with value < A_j
        if idx_j > 0: # Only need to query if idx_j > 0 (i.e., A_j is not the smallest possible value)
             # Query the tree state *before* processing A_j
             k = query(roots[-1], 0, M - 1, 0, idx_j - 1)

        # The length of the Longest Increasing Subsequence ending specifically with A_j at index j is k + 1.
        lis_len_ending_at_Aj = k + 1

        # --- Update Step ---
        # Update the segment tree at position idx_j with this new LIS length `lis_len_ending_at_Aj`.
        # This `update` call creates a new tree version representing the state after processing A[0...j].
        # `new_root` is the root of this new version (version T_{j+1}).
        new_root = update(roots[-1], 0, M - 1, idx_j, lis_len_ending_at_Aj)
        # Store the root of the new version. `roots[j+1]` will hold this root.
        roots.append(new_root) 

    # --- Answering Queries ---
    results_output = [] # Collect results as strings for efficient batch printing at the end.
    for R_i, X_i in queries_data:
        # For a query (R_i, X_i), we need the state of the LIS computation after processing the prefix A[0...R_i-1].
        # This state is captured by the segment tree version rooted at `roots[R_i]`.
        
        # Find the rank `k` corresponding to the largest value in `coords` that is less than or equal to X_i.
        # `bisect_right(coords, X_i)` gives the count of elements in `coords` that are <= X_i.
        # The index (rank) of the largest element <= X_i is this count minus 1.
        k = bisect_right(coords, X_i) - 1
        
        ans = 0 # Initialize answer for the query
        # Based on problem constraints (X_i >= min(A_1..A_{R_i})), `k` should always be >= 0.
        # This guarantees that the query range [0, k] is valid.
        if k >= 0: # Add a defensive check anyway.
            # Query the tree version `roots[R_i]` for the maximum value in the range [0, k].
            # This range includes all ranks corresponding to values <= X_i.
            # The maximum value found is the length of the LIS satisfying the query conditions.
            ans = query(roots[R_i], 0, M - 1, 0, k)
        # else: ans remains 0. This case should not happen under given problem constraints.
        
        # Append the computed answer (as a string) to the list for final output.
        results_output.append(str(ans))

    # Print all results, each on a new line. Using join is generally faster than multiple print calls.
    print('
'.join(results_output))

# Execute the main solver function when the script is run.
solve()