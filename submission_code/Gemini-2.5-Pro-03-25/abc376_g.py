# YOUR CODE HERE
import sys
import heapq

# Increase recursion depth limit for deep trees.
# The constraints state N <= 2e5 and Sum of N over test cases <= 2e5.
# A single test case could have N up to 2e5, requiring a deep recursion stack.
# Set a limit slightly larger than the maximum possible N.
try:
    # Standard practice in competitive programming for large N on platforms like Codeforces/AtCoder
    sys.setrecursionlimit(2 * 10**5 + 50) 
except OverflowError: 
    # Fallback if the system restricts setting a very high limit.
    # Use a reasonably large value and hope it suffices.
     sys.setrecursionlimit(2 * 10**5 + 10) 

MOD = 998244353

def fast_pow(base, power):
    """
    Computes (base^power) % MOD efficiently using binary exponentiation.
    """
    result = 1
    while power > 0:
        if power % 2 == 1:
            result = (result * base) % MOD
        base = (base * base) % MOD
        power //= 2
    return result

def inverse(a):
    """
    Computes the modular multiplicative inverse of a modulo MOD using Fermat's Little Theorem.
    Requires MOD to be a prime number, which 998244353 is.
    Assumes a is not divisible by MOD. The problem guarantees Q not divisible by MOD,
    and Q corresponds to the total sum S here.
    """
    # Check if a is 0 mod MOD? Problem statement guarantees S != 0 mod MOD.
    # Also, constraints state S = sum a_i <= 10^8 < MOD, so S % MOD = S.
    return fast_pow(a, MOD - 2)

# Custom class to represent items in the priority queue.
# Objects of this class will store node index, its subtree weight sum A, and subtree size Size.
# The comparison logic (__lt__) is implemented to work with Python's heapq module (min-heap)
# such that popping from the heap yields the node with the maximum priority.
class NodePriority:
    def __init__(self, node_idx, A, Size):
        self.node_idx = node_idx
        # Store exact integer values for A and Size. A can potentially exceed MOD,
        # but needs full value for accurate priority comparison based on ratio A/Size.
        self.A = A 
        self.Size = Size

    def __lt__(self, other):
        """
        Less-than comparison method for heapq. Since heapq is a min-heap,
        an element 'self' is considered "less than" element 'other' if 'self'
        should be popped *earlier* than 'other'. This means 'self' must have
        a higher priority.
        Priority is defined as A/Size ratio. Higher ratio means higher priority.
        Ties are broken by smaller node index (smaller index gets higher priority).

        Comparison Logic:
        Priority(self) > Priority(other) if:
        1. self.A / self.Size > other.A / other.Size
           (Equivalent to self.A * other.Size > other.A * self.Size to avoid division/floats)
        2. Ratios are equal AND self.node_idx < other.node_idx
        """
        
        # Compare ratios using cross-multiplication
        val1 = self.A * other.Size
        val2 = other.A * self.Size
        
        if val1 != val2:
            # If val1 > val2, 'self' has a higher A/Size ratio, thus higher priority.
            # For min-heap, higher priority means "less than".
            return val1 > val2 
        else:
            # Ratios are equal. Tie-break using node index.
            # Smaller index means higher priority.
            # If self.node_idx < other.node_idx, 'self' has higher priority.
            return self.node_idx < other.node_idx

def solve():
    # Read the size of the tree (N+1 vertices, N non-root vertices)
    N = int(sys.stdin.readline())
    
    # Read the parent array p_1, ..., p_N
    parents = list(map(int, sys.stdin.readline().split()))
    # Read the weights a_1, ..., a_N
    a = list(map(int, sys.stdin.readline().split()))
    
    # Adjust the weights array 'a' to be 1-indexed for convenience.
    # a_adjusted[i] will store the weight for vertex i. a_adjusted[0] is unused/0.
    a_adjusted = [0] * (N + 1)
    total_S = 0 # Calculate the total sum S = sum(a_i) for i=1..N
    for i in range(N):
        a_adjusted[i+1] = a[i]
        # Total sum S constraints: S >= 1, S <= 10^8. S fits in standard integer types.
        total_S += a[i] 

    # Build the adjacency list representation of the tree.
    # adj[u] will contain a list of children of vertex u.
    adj = [[] for _ in range(N + 1)]
    for i in range(N): # Vertex i+1 has parent parents[i]
        p_i = parents[i]
        adj[p_i].append(i + 1)

    # Arrays to store subtree aggregate values computed by DFS.
    # subtree_A[u]: sum of weights 'a' for all nodes in the subtree rooted at u.
    # subtree_Size[u]: number of nodes in the subtree rooted at u.
    subtree_A = [0] * (N + 1) 
    subtree_Size = [0] * (N + 1) 

    # Define the recursive Depth First Search function to compute subtree aggregates.
    def dfs_compute_stats(u):
        """
        Performs DFS starting from node u. Computes and stores the total weight sum (A)
        and total number of nodes (Size) for the subtree rooted at u.
        Uses post-order traversal logic: compute for children first, then combine for parent.
        """
        current_size = 1 # Count node u itself
        current_A = a_adjusted[u] # Weight of node u (0 if u=0)
                
        # Recursively call for children
        for v in adj[u]:
            dfs_compute_stats(v) # Compute stats for child subtree first
            # Aggregate results from child subtree
            current_size += subtree_Size[v] 
            current_A += subtree_A[v] 
        
        # Store the computed aggregates for node u
        subtree_Size[u] = current_size
        subtree_A[u] = current_A

    # Start the DFS from the root node 0 to compute aggregates for all nodes.
    dfs_compute_stats(0)

    # Handle edge case where S=0, though constraints state a_i >= 1, so S >= 1.
    if total_S == 0:
      print(0)
      return

    # Initialize the priority queue (min-heap) using Python's heapq module.
    # It will store NodePriority objects.
    pq = [] 

    # Initialize the search frontier by adding all children of the root node 0.
    for child in adj[0]:
        # Create a NodePriority object for each child and push it onto the heap.
        # The priority is determined by subtree_A and subtree_Size of the child.
        heapq.heappush(pq, NodePriority(child, subtree_A[child], subtree_Size[child]))

    # Accumulator for the numerator of the expected value calculation.
    # This is Sum(a_i * k_i) where k_i is the rank (operation number) when node i is searched.
    total_expected_sum_numerator = 0
    k = 0 # Operation count, starts at 0, increments for each search operation.

    # Main simulation loop: Continues as long as there are nodes in the frontier (priority queue).
    while pq:
        # Extract the node with the highest priority from the heap.
        # Due to the custom __lt__ method, heappop() returns the node with max A/Size ratio.
        best_node_priority = heapq.heappop(pq)
        u = best_node_priority.node_idx # Get the index of the node to search
        
        k += 1 # Increment the operation count (this is the k-th node searched)

        # Add the contribution of searching node u to the total sum numerator.
        # Contribution = (weight of node u) * (operation number k). Compute modulo MOD.
        term = (a_adjusted[u] * k) % MOD
        total_expected_sum_numerator = (total_expected_sum_numerator + term) % MOD
        
        # Add all children of the newly searched node u into the priority queue.
        # These children are now eligible to be searched in future steps.
        for v in adj[u]:
             heapq.heappush(pq, NodePriority(v, subtree_A[v], subtree_Size[v]))

    # After the loop, total_expected_sum_numerator holds Sum(a_i * k_i) mod MOD.
    # The final expected value is (Numerator / S) mod MOD. Compute S^{-1} mod MOD.
    S_inv = inverse(total_S) # Calculate modular inverse of total sum S.
    
    # Final answer = (Numerator * S_inv) % MOD
    final_ans = (total_expected_sum_numerator * S_inv) % MOD
    print(final_ans)

# Read the number of test cases T
T = int(sys.stdin.readline())
# Process each test case independently
for _ in range(T):
    solve()