# YOUR CODE HERE
import sys

# Use iterative find to avoid recursion depth limits typically associated with recursive implementations
def find_iterative(i, parent):
    """Find the root of the set containing element i using path compression.
    
    Args:
        i (int): The element whose set root needs to be found.
        parent (list): The parent array for the DSU structure.
        
    Returns:
        int: The root of the set containing element i.
    """
    root = i
    # First pass: Find the root of the tree containing i
    while parent[root] != root:
        root = parent[root]
    
    # Second pass: Path compression. Make all nodes on the path from i to root
    # point directly to the root. This flattens the tree structure for future queries.
    curr = i
    while parent[curr] != root:
        next_node = parent[curr] # Store the next node in the path
        parent[curr] = root      # Set current node's parent directly to root
        curr = next_node         # Move to the next node in the original path
    return root

def merge_lists(list1, list2, k_max):
    """Merges two sorted lists (descending) into a single sorted list (descending),
       keeping only unique elements and limiting the size to k_max.
       
    Args:
        list1 (list): First sorted list of integers (descending).
        list2 (list): Second sorted list of integers (descending).
        k_max (int): The maximum number of elements to keep in the merged list.
        
    Returns:
        list: A new sorted list (descending) containing unique elements from both inputs,
              truncated to at most k_max elements.
    """
    merged = [] # The resulting merged list
    p1, p2 = 0, 0 # Pointers for list1 and list2 respectively
    
    # Iterate while the merged list size is less than k_max 
    # and there are still elements in at least one of the input lists
    while len(merged) < k_max and (p1 < len(list1) or p2 < len(list2)):
        # Get current elements from both lists. Use -1 as a sentinel value if a list is exhausted.
        # Vertices are numbered 1 to N, so -1 is safe to use as it's smaller than any valid vertex number.
        val1 = list1[p1] if p1 < len(list1) else -1
        val2 = list2[p2] if p2 < len(list2) else -1

        # Determine the larger candidate element to potentially add to the merged list
        if val1 >= val2:
            candidate = val1
            if candidate != -1: # Process only valid vertex IDs (must be >= 1)
                # Add candidate if the merged list is empty or the candidate is different from the last added element (ensures uniqueness)
                if not merged or merged[-1] != candidate:
                     merged.append(candidate)
            p1 += 1 # Move pointer for list1
            # If the values were equal and val2 is a valid vertex ID, advance p2 as well.
            # This correctly handles cases where the same element appears in both lists.
            if val1 == val2 and val2 != -1:
                 p2 += 1
        else: # val2 > val1
            candidate = val2
            # Since val2 > val1 and val1 >= -1, candidate must be a valid vertex ID (>=0, actually >=1).
            # Add candidate if the merged list is empty or the candidate is different from the last added element.
            if not merged or merged[-1] != candidate:
                 merged.append(candidate)
            p2 += 1 # Move pointer for list2
            
    return merged


def union(u, v, parent, size, top_elements, k_max):
    """Merges the sets containing elements u and v using the union-by-size heuristic.
       It also merges the lists of top k elements associated with the roots of the sets.
       
    Args:
        u (int): First element.
        v (int): Second element.
        parent (list): Parent array for DSU.
        size (list): Size array for DSU (stores size of components).
        top_elements (list): A list where top_elements[root] stores the sorted list of top k elements for the component rooted at 'root'.
        k_max (int): The maximum number of top elements to track per component.
    """
    root_u = find_iterative(u, parent) # Find root of u's component
    root_v = find_iterative(v, parent) # Find root of v's component
    
    if root_u != root_v: # Only proceed if u and v are in different components
        # Union by size heuristic: attach the smaller tree to the root of the larger tree
        if size[root_u] < size[root_v]:
            # Swap roots so that root_u is always the root of the larger (or equal size) component
            root_u, root_v = root_v, root_u 
        
        # Merge component root_v into component root_u
        parent[root_v] = root_u             # Set parent of root_v to root_u
        size[root_u] += size[root_v]        # Update size of the merged component
        
        # Merge the lists of top k elements
        new_top_k = merge_lists(top_elements[root_u], top_elements[root_v], k_max)
        top_elements[root_u] = new_top_k # Update the list for the new root
        
        # Clear the list for the merged root (root_v), as it's no longer a root.
        # This is optional but can help manage memory. Check index bounds for safety.
        if root_v < len(top_elements): 
             top_elements[root_v] = [] 


def solve():
    # Read number of vertices N and number of queries Q
    N, Q = map(int, sys.stdin.readline().split())
    
    # Initialize Disjoint Set Union (DSU) structure
    # parent[i] stores the parent of node i. Initially, each node is its own parent.
    parent = list(range(N + 1)) 
    # size[i] stores the size of the component rooted at i. Initially, all sizes are 1.
    size = [1] * (N + 1) 
    
    # Initialize storage for top k elements for each component.
    # Vertices are 1-based indexed. We use a list of lists.
    # top_elements[i] will store the list of top elements for the component rooted at i.
    # Initially, each vertex is in its own component, and its list contains only itself.
    # We add a dummy empty list at index 0 to align indices with vertex numbers (1 to N).
    top_elements = [[]] + [[i] for i in range(1, N + 1)] 
    
    # K_max is the maximum value of k specified in the problem constraints.
    # We need to track up to K_max largest elements per component.
    K_max = 10 
    
    results = [] # List to store answers for Type 2 queries

    # Process Q queries
    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        
        q_type = query[0] # Query type (1 or 2)
        
        if q_type == 1:
            # Type 1 query: Add an edge between u and v
            u, v = query[1], query[2]
            # Perform union operation to merge components if u and v are not already connected
            union(u, v, parent, size, top_elements, K_max)
        else: # q_type == 2
            # Type 2 query: Find k-th largest vertex connected to v
            v, k = query[1], query[2]
            # Find the representative (root) of the component containing v
            root_v = find_iterative(v, parent) 
            
            # Retrieve the list of top k elements stored for this component's root
            current_top_k = top_elements[root_v] 
            
            # Check if the component has at least k elements
            if k > len(current_top_k):
                # If k is larger than the number of elements stored, the k-th largest doesn't exist
                results.append("-1") 
            else:
                # The k-th largest element is at index k-1 in the 0-indexed sorted list
                results.append(str(current_top_k[k-1]))

    # Print all collected results for Type 2 queries, each on a new line
    print("
".join(results))

# Execute the main solve function when the script is run
solve()