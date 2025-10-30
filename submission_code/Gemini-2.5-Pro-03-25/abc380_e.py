# YOUR CODE HERE
import sys
from collections import defaultdict

# Set higher recursion depth limit for safety, although iterative find should avoid issues.
# Using iterative find which doesn't rely on deep recursion stack.
# sys.setrecursionlimit(2000000) 

# Global DSU state arrays declaration. They will be initialized inside solve().
parent = []
set_size = []
segment_start = []
segment_end = []
segment_color = []

# Iterative find operation with path compression for DSU
def find(i):
    """Find the root of the set containing element i, with path compression."""
    global parent # Indicate usage of the global parent list
    root = i
    # Traverse up the parent pointers to find the root of the set
    while parent[root] != root:
        root = parent[root]
    
    # Path compression: Iterate from i up to the root again, 
    # making each node point directly to the root.
    curr = i
    while parent[curr] != root:
        next_node = parent[curr] # Store the next node in the path
        parent[curr] = root      # Make current node point directly to the root
        curr = next_node         # Move to the next node
    return root


# Union operation using union by size heuristic for DSU
def union(i, j):
    """Merge the sets containing elements i and j, using union by size."""
    # Indicate usage of global state arrays
    global parent, set_size, segment_start, segment_end, segment_color
    
    root_i = find(i) # Find the root of the set containing i
    root_j = find(j) # Find the root of the set containing j
    
    if root_i != root_j: # Only merge if they are in different sets
        # Union by size heuristic: Attach the root of the smaller tree 
        # to the root of the larger tree.
        if set_size[root_i] < set_size[root_j]:
            # Swap roots so that root_i always refers to the root of the larger (or equal size) set
            root_i, root_j = root_j, root_i 
        
        # Merge set j (rooted at root_j) into set i (rooted at root_i)
        parent[root_j] = root_i # Make root_j a child of root_i
        set_size[root_i] += set_size[root_j] # Update the size of the merged set
        
        # Update segment boundaries for the merged set, now represented by root_i
        # The start index is the minimum of the start indices of the two merged segments
        segment_start[root_i] = min(segment_start[root_i], segment_start[root_j])
        # The end index is the maximum of the end indices of the two merged segments
        segment_end[root_i] = max(segment_end[root_i], segment_end[root_j])
        # The color attribute segment_color[root_i] remains unchanged because we only merge segments
        # if their colors match the target color 'c' after the central segment is repainted.
        # The color of root_j becomes irrelevant as it's no longer a root.
        
        return True # Indicates that a merge operation was performed
    return False # Indicates that i and j were already in the same set, no merge needed

# Main function to process inputs and queries
def solve():
    # Indicate intent to initialize/use global DSU state arrays within this function
    global parent, set_size, segment_start, segment_end, segment_color 

    # Read N (number of cells) and Q (number of queries)
    N, Q = map(int, sys.stdin.readline().split())

    # Initialize DSU structure for N cells. We use 1-based indexing for cells.
    # Size of arrays is N+1 to accommodate indices 1 through N. Index 0 is unused.
    parent = list(range(N + 1)) # Each element is initially its own parent
    set_size = [1] * (N + 1)    # Each set initially has size 1
    segment_start = list(range(N + 1)) # Start index of segment represented by element i is initially i
    segment_end = list(range(N + 1))   # End index of segment represented by element i is initially i
    # Initially, cell i has color i. This color is stored at the root of its set.
    segment_color = list(range(N + 1)) 

    # Dictionary to keep track of the total count of cells for each color.
    # Using defaultdict(int) means accessing a non-existent key returns 0.
    counts = defaultdict(int)
    for i in range(1, N + 1):
        counts[i] = 1 # Each cell starts with a unique color, so count for color i is 1.

    results = [] # List to store answers for type 2 queries
    
    # Process each query
    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        
        query_type = query[0] # Get the query type (1 or 2)

        if query_type == 1: # Type 1: Repaint query
            x, c = query[1], query[2] # Get cell index x and target color c
            
            root_x = find(x) # Find the representative (root) of the set containing cell x
            k = segment_color[root_x] # Get the current color of the segment containing x
            
            if k == c:
                continue # If the segment is already painted with color c, nothing to do.
                
            # Get properties of the segment before potential merges
            current_segment_size = set_size[root_x] # Size of the segment to be repainted
            L = segment_start[root_x] # Start index of the segment
            R = segment_end[root_x]   # End index of the segment
            
            # Update the global counts of cells for the old color k and the new color c
            counts[k] -= current_segment_size
            if counts[k] == 0:
                 # Optional cleanup: remove color k from counts if its count drops to zero
                 # This might save memory if there are many color changes leading to empty colors.
                 if k in counts: 
                    del counts[k] 
            counts[c] += current_segment_size
            
            # Repaint the segment to color c by updating the color attribute stored at its root
            segment_color[root_x] = c
            
            # Check the left adjacent cell (cell L-1) if it exists (L > 1)
            if L > 1:
                root_left = find(L - 1) # Find the root of the segment to the left
                # Check if the left segment now matches the new color c
                if segment_color[root_left] == c:
                     # If colors match, merge the current segment (containing x) with the left segment
                     if union(x, L - 1): 
                         # A merge occurred. The root of x might have changed.
                         # Re-find the root and update L, R based on the potentially new root's state.
                         # This ensures the correct segment boundaries are used for the right check.
                         root_x = find(x) 
                         L = segment_start[root_x] 
                         R = segment_end[root_x]

            # Check the right adjacent cell (cell R+1) if it exists (R < N)
            # Use potentially updated root_x, L, R from the left merge check.
            if R < N:
                root_right = find(R + 1) # Find the root of the segment to the right
                # Check if the right segment matches the new color c
                if segment_color[root_right] == c:
                     # If colors match, merge the current segment (containing x) with the right segment
                     union(x, R + 1) 
                     # The state of root_x, L, R might change again after this merge, but it's the
                     # end of processing for this type 1 query, so no need to update these variables further.

        else: # query_type == 2: Count query
            c = query[1] # Get the color c to count
            # Append the count of cells with color c to the results list.
            # Accessing counts[c] will return 0 via defaultdict if color c currently has no cells.
            results.append(counts[c])

    # After processing all queries, print the collected results for type 2 queries
    for res in results:
        print(res)

# Execute the main function
solve()