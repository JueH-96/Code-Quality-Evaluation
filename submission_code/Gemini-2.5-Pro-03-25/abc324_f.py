# YOUR CODE HERE
import sys

# Function to read input faster
def get_ints():
    """Reads a line from stdin, removes leading/trailing whitespace, 
    splits by space, and converts parts to integers."""
    return map(int, sys.stdin.readline().strip().split())

def solve():
    """Solves the problem: finds the maximum ratio of total beauty to total cost
    for a path from vertex 1 to vertex N in a directed acyclic graph."""
    
    N, M = get_ints()
    
    # Adjacency list representation of the graph.
    # adj[u] contains a list of tuples (v, beauty, cost) for edges starting from u.
    # We use 1-based indexing for vertices (indices 1 to N) to match the problem statement.
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v, b, c = get_ints()
        # The problem statement guarantees u < v. This implies the graph is a Directed Acyclic Graph (DAG).
        # The vertices are naturally topologically sorted by their indices 1, 2, ..., N.
        adj[u].append((v, b, c))

    # This function checks if there exists a path P from vertex 1 to vertex N
    # such that the ratio Sum(beauty_i for i in P) / Sum(cost_i for i in P) is at least x.
    # This is equivalent to checking if the maximum path weight is non-negative,
    # where edge weights are defined as (beauty - x * cost).
    def check(x):
        """Checks if a ratio of at least x is achievable."""
        
        # dp[v] stores the maximum path weight (sum of beauty - x * cost) from vertex 1 to vertex v.
        # Initialize all dp values to negative infinity, representing unreachability or minimum possible value.
        dp = [float('-inf')] * (N + 1)
        
        # The path weight from vertex 1 to itself is 0 (empty path).
        dp[1] = 0.0
        
        # Iterate through vertices in topological order. Since u < v for all edges,
        # the natural order 1, 2, ..., N is a valid topological sort.
        for v_curr in range(1, N + 1):
            # If the current vertex v_curr is unreachable from vertex 1, we cannot extend paths from it.
            # dp[v_curr] being -inf indicates unreachability.
            if dp[v_curr] == float('-inf'):
                continue
            
            # Explore all outgoing edges from the current vertex v_curr
            for v_next, beauty, cost in adj[v_curr]:
                # Calculate the weight of the edge (v_curr, v_next) for the given target ratio x.
                # Convert beauty and cost to float for accurate floating-point arithmetic.
                weight = float(beauty) - x * float(cost)
                
                # Try to relax the edge: if the path to v_next via v_curr has a higher weight
                # than the current best known path weight to v_next, update dp[v_next].
                # This is the standard relaxation step for finding the longest path in a DAG.
                if dp[v_curr] + weight > dp[v_next]:
                     dp[v_next] = dp[v_curr] + weight

        # After iterating through all vertices, dp[N] holds the maximum path weight from 1 to N.
        # The problem guarantees that a path from 1 to N exists, so dp[N] will eventually become finite.
        # If the maximum path weight is non-negative (>= 0), it means a ratio of at least x is achievable.
        # We compare against 0.0. Small floating point inaccuracies near 0 are possible, but usually handled fine by binary search.
        return dp[N] >= 0.0

    # We perform binary search on the possible values of the ratio B/C.
    # The ratio must be non-negative since beauty and cost are positive.
    low = 0.0  
    
    # The maximum possible ratio for a single edge is max_beauty / min_cost = 10000 / 1 = 10000.
    # The overall path ratio is bounded. We use a slightly larger upper bound for safety.
    high = 10001.0 

    # Perform a fixed number of iterations for the binary search. 100 iterations are typically sufficient
    # for competitive programming problems requiring high precision (like 10^-9).
    # Each iteration halves the search interval, reducing the interval size by a factor of 2^100.
    for _ in range(100):
        mid = (low + high) / 2.0
        # Check if a ratio of `mid` is achievable using the check function.
        if check(mid):
            # If `mid` is achievable, it means the optimal ratio could be `mid` or potentially higher.
            # So, we update `low` to `mid` to search in the upper half [mid, high].
            low = mid
        else:
            # If `mid` is not achievable, the optimal ratio must be less than `mid`.
            # Update `high` to `mid` to search in the lower half [low, mid].
            high = mid
            
    # After 100 iterations, `low` will be a very close approximation of the maximum possible ratio.
    # Print the result formatted to 15 decimal places to ensure sufficient precision according to the problem statement
    # (required precision is 10^-9, 15 decimal places provide much more).
    print(f"{low:.15f}")

# Execute the main function to solve the problem when the script is run.
if __name__ == '__main__':
    solve()