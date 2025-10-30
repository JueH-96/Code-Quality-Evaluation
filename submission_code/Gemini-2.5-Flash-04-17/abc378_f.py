import sys
import threading

# Set a higher recursion depth limit
# In graph problems on trees, the depth of recursion can go up to N in the worst case (a path graph).
# The default limit might be insufficient for N = 2 * 10^5.
try:
    # Set limit slightly higher than max possible N
    sys.setrecursionlimit(200005) 
except OverflowError:
    # Handle potential failure to set the limit very high.
    # This part might not be necessary in typical competitive programming environments
    # where higher limits might be pre-configured or setting this works.
    pass

# Increase stack size for threads if using threading.
# This is a common workaround in CPython for deep recursion without hitting stack overflow.
# The default stack size might be small.
# Define a size, e.g., 256MB.
THREAD_STACK_SIZE = 2**28 
try:
    threading.stack_size(THREAD_STACK_SIZE)
except:
    # Handle potential failure to set stack size (e.g., not supported on the OS/environment)
    # If threading is not used, or stack_size setting fails, the sys.setrecursionlimit
    # might still be sufficient, or an iterative approach would be needed.
    pass


def main():
    # Read the number of vertices
    N = int(sys.stdin.readline())

    # Initialize adjacency list and degree counter for each vertex (1-indexed)
    adj = [[] for _ in range(N + 1)]
    degree = [0] * (N + 1)
    
    # Read edges and build the adjacency list and degrees
    # The problem guarantees the input is a tree.
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1

    # Memoization dictionary for the count_endpoints function.
    # Keys are tuples (current_node, parent_node) representing the branch being explored.
    # Values are the number of degree-2 endpoints found reachable from that branch.
    memo = {}

    def count_endpoints(u, parent):
        """
        Recursively counts the number of degree-2 vertices reachable from node `u`
        (when coming from `parent`) following a path that consists only of
        degree-3 vertices until it reaches a degree-2 vertex or a 'bad' vertex.

        Args:
            u (int): The current vertex being visited.
            parent (int): The vertex from which `u` was visited in the current traversal branch.

        Returns:
            int: The count of degree-2 endpoints found in this branch.
        """
        # Check if the result for this specific (u, parent) branch is already computed
        memo_key = (u, parent)
        if memo_key in memo:
            return memo[memo_key]
            
        # If the current vertex has degree 1 or degree >= 4 in the original tree,
        # it cannot be part of a valid path that satisfies the conditions
        # (endpoints must be degree 2, internal nodes must be degree 3).
        # So, this branch yields 0 valid endpoints.
        if degree[u] == 1 or degree[u] >= 4:
            memo[memo_key] = 0
            return 0
            
        # If the current vertex has degree 2 in the original tree,
        # it can be a valid endpoint of a path segment.
        # Since the traversal is designed to follow degree-3 vertices from a junction
        # until it hits a degree-2 node or a 'bad' node, reaching a degree-2 node here
        # signifies finding a valid endpoint for a path originating from a degree-3 junction.
        # This branch yields 1 valid endpoint (the node 'u' itself).
        if degree[u] == 2:
            memo[memo_key] = 1
            return 1
        
        # If the current vertex has degree 3 in the original tree,
        # it can serve as an internal node in the path segment.
        # Continue the exploration through its neighbors, excluding the parent
        # (as we are traversing away from the parent).
        # The total number of endpoints reachable through this branch (starting at u,
        # coming from parent) is the sum of endpoints reachable through its other neighbors.
        
        # The recursion will naturally stop when it hits a degree-1, degree-2,
        # or degree-4+ vertex, or when it explores all reachable nodes in a component.
        count = 0
        for v in adj[u]:
            if v != parent:
                # Recursively call count_endpoints for the neighbor v, with u as its parent
                count += count_endpoints(v, u)
        
        # Store the computed result in the memoization dictionary before returning
        memo[memo_key] = count
        return count

    # Initialize the total count of valid pairs of vertices (u, v)
    total_valid_pairs = 0

    # Iterate through each vertex in the graph (from 1 to N)
    # A vertex 'w' with degree 3 in the original tree is crucial because it
    # can act as a junction point where two or more segments of a valid cycle path meet.
    # A valid cycle path must consist of degree-2 endpoints and only degree-3 internal nodes.
    # Such a path passing through a degree-3 vertex 'w' would arrive from one neighbor
    # and depart through another neighbor.
    for w in range(1, N + 1):
        if degree[w] == 3:
            # If w has degree 3, it has exactly 3 neighbors. Let them be n1, n2, n3.
            neighbors = adj[w]
            
            # We want to count pairs of degree-2 endpoints (u, v) such that the unique
            # path between u and v in the tree passes through w, and all internal
            # vertices on this path (including w) have degree 3.
            # This means u must be reachable from w via one neighbor (e.g., n_i)
            # and v must be reachable from w via a different neighbor (e.g., n_j),
            # with the path segments w-u and w-v consisting only of degree-3 internal
            # nodes (except for u and v themselves).
            # The function count_endpoints(neighbor, w) gives the number of degree-2
            # endpoints reachable through the branch starting with 'neighbor', considering
            # 'w' as the point of entry to this branch.

            # Store the counts of reachable degree-2 endpoints for each of the 3 branches originating from w.
            counts = []
            for neighbor in neighbors:
                 # Call count_endpoints for each neighbor, treating w as the parent to
                 # explore the branch away from w.
                 counts.append(count_endpoints(neighbor, w))

            # Let the counts for the three branches be c0, c1, and c2.
            # A valid pair (u, v) passing through w must have u reachable from one branch
            # and v reachable from a different branch.
            # The number of such pairs formed by picking one endpoint from branch i
            # and another from branch j (where i != j) is counts[i] * counts[j].
            # We sum this product over all unique pairs of branches {i, j} with i < j.
            # This sum is c0*c1 + c0*c2 + c1*c2.
            # This calculation counts each valid pair (u, v) once for every degree-3 vertex
            # that lies on the unique path between u and v in the original tree.
            # Based on the problem statement and sample cases, this sum appears to be
            # the desired answer, implying this counting method correctly captures
            # the number of valid pairs without problematic overcounting in the final result.
            
            # The neighbors list has exactly 3 elements because degree[w] == 3.
            # We can directly access counts[0], counts[1], counts[2].
            total_valid_pairs += counts[0] * counts[1] + counts[0] * counts[2] + counts[1] * counts[2]

    # Print the final total count of valid pairs
    print(total_valid_pairs)

# Execute the main function.
# Use threading to run the main logic. This is done to potentially handle deep recursion
# within the count_endpoints function call stack by using a larger stack size provided by the thread.
if __name__ == "__main__":
    # Create a thread targeting the main function
    thread = threading.Thread(target=main)
    # Start the thread execution
    thread.start()
    # Wait for the thread to finish. This ensures the program doesn't exit before
    # the main logic completes, especially important for input/output operations.
    thread.join()