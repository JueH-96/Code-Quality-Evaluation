# YOUR CODE HERE
import sys
from collections import Counter

# Set higher recursion depth for deep DSU paths. 
# This might be needed for large N where the component structure could lead to deep recursion.
# The judge environment might have different limits, but locally this helps avoid RecursionError.
try:
    # Check if we are in an environment where we can set recursion depth
    # Some environments like online judges might restrict this
    sys.setrecursionlimit(300000) 
except Exception as e:
    # If setting recursion limit fails, it's usually fine on judge systems with higher default limits.
    pass 


# DSU (Disjoint Set Union) Find operation with path compression
# Finds the representative (root) of the set containing element v
# parent: list where parent[i] is the parent of node i
def find_set_dsu(v, parent):
    # If v is the root of its tree, return v
    if v == parent[v]:
        return v
    # Path compression: Make nodes point directly to the root
    parent[v] = find_set_dsu(parent[v], parent)
    return parent[v]

# DSU Union operation using union by rank
# Merges the sets containing elements a and b
# parent: list of parents
# rank: list storing the rank (approximate height) of the tree rooted at i
def unite_sets_dsu(a, b, parent, rank):
    # Find roots of the sets containing a and b
    a = find_set_dsu(a, parent)
    b = find_set_dsu(b, parent)
    
    # If a and b are already in the same set, do nothing
    if a != b:
        # Union by rank heuristic: attach the smaller rank tree under the larger rank tree
        if rank[a] < rank[b]:
            a, b = b, a # Swaps a and b to ensure rank[a] >= rank[b]
        parent[b] = a # Attach tree b to tree a
        # If ranks were equal, the rank of the new root increases by 1
        if rank[a] == rank[b]:
            rank[a] += 1
        return True # Return True indicating a merge happened
    return False # Return False indicating already in same set


# Function to find the next active index >= i using path compression on next_idx_arr
# Active index `k` means `next_idx_arr[k] == k`. It has not been merged rightwards yet.
# next_idx_arr[i] stores the representative of the block of indices starting at i that have been merged together.
# This optimization helps to efficiently skip over indices already merged into components.
def find_next(i, N, next_idx_arr):
    # Base case: if i is out of bounds (N is used as a sentinel)
    if i >= N : return N 
    
    # If i is the representative of its block (not merged rightward)
    if next_idx_arr[i] == i:
        return i
    
    # Path compression: find the ultimate representative and update next_idx_arr[i]
    res = find_next(next_idx_arr[i], N, next_idx_arr)
    next_idx_arr[i] = res 
    return res


# Main logic to solve a single test case
def solve():
    # Read N (length of sequences) and K (distance constraint)
    N, K = map(int, sys.stdin.readline().split())
    # Read sequence A
    A = list(map(int, sys.stdin.readline().split()))
    # Read sequence B
    B = list(map(int, sys.stdin.readline().split()))

    # Initialize DSU structures for main components (indices 0 to N-1)
    parent_dsu = list(range(N))
    rank_dsu = [0] * N
    
    # Initialize next_idx structure for optimization: size N+1. Indices 0..N-1 used, N is sentinel.
    # Initially, each index i points to itself, meaning it's active.
    next_idx_arr = list(range(N + 1)) 

    # Build connected components using optimized DSU
    # Iterate through each index i from 0 to N-1
    for i in range(N):
        # Find the next active index `curr` starting from i+1
        curr = find_next(i + 1, N, next_idx_arr)
        
        # Iterate through active indices `curr` within the range [i+1, i+K]
        while curr < N and curr <= i + K:
            # Unite index `i` and index `curr` into the same component using main DSU
            unite_sets_dsu(i, curr, parent_dsu, rank_dsu)
            
            # Mark `curr` as processed (merged into the component containing i).
            # Update its next_idx pointer to point to the next active index after `curr`.
            # This allows future find_next calls to efficiently skip over `curr`.
            next_active_after_curr = find_next(curr + 1, N, next_idx_arr)
            next_idx_arr[curr] = next_active_after_curr
            
            # Move to the next potentially active index.
            # After updating next_idx_arr[curr], find_next(curr, ...) will effectively jump
            # to the next active index using path compression.
            curr = find_next(curr, N, next_idx_arr)


    # Group indices by component root after all unions are done
    components = {} # Dictionary to store lists of indices for each component root
    for i in range(N):
        # Find the root of the component containing index i
        root = find_set_dsu(i, parent_dsu)
        # Add index i to the list associated with its root
        if root not in components:
            components[root] = []
        components[root].append(i)

    # Check if the transformation is possible based on multiset equality per component
    possible = True
    for root in components: # Iterate through each found component
        component_indices = components[root] # Get list of indices in this component
        
        # Compute the multiset of values from sequence A for this component
        multiset_A = Counter()
        for idx in component_indices:
            multiset_A[A[idx]] += 1
        
        # Compute the multiset of values from sequence B for this component
        multiset_B = Counter()
        for idx in component_indices:
            multiset_B[B[idx]] += 1
        
        # If the multisets for A and B within this component are not identical,
        # the transformation is impossible.
        if multiset_A != multiset_B:
            possible = False
            break # No need to check other components

    # Print the result for this test case
    if possible:
        print("Yes")
    else:
        print("No")

# Read the number of test cases
T = int(sys.stdin.readline())
# Process each test case
for _ in range(T):
   solve()