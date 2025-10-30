# YOUR CODE HERE
import sys

# Set a higher recursion depth limit. Although N=18 means a tree depth of N+1=19, 
# which is well within Python's default limit (~1000), complex interactions or
# specific test cases might benefit from a higher limit.
# Competitive programming platforms often have higher default limits anyway.
# This is more of a precaution.
try:
    # Increase recursion depth limit
    sys.setrecursionlimit(2000) 
except Exception as e:
    # Some environments might restrict changing the recursion limit.
    # We can print an error message to stderr for debugging if needed, but continue execution.
    # print(f"Warning: Could not set recursion depth limit: {e}", file=sys.stderr)
    pass

def query_judge(i, j):
    """ 
    Sends a query formatted as '? i j' to the judge via standard output,
    ensures the output buffer is flushed, then reads the integer response T 
    from standard input.
    If the judge responds with T = -1 (indicating an error like invalid query parameters, 
    exceeding the query limit, etc.), the program terminates immediately as per the 
    problem specification. Otherwise, returns the integer T.
    """
    print(f"? {i} {j}", flush=True)
    T = int(sys.stdin.readline())
    if T == -1:
        # Judge signaled an error condition. Terminate the program immediately.
        sys.exit() 
    return T

# Memoization dictionary to store results of queries (i, j) -> sum T.
# This avoids asking the judge the same question multiple times, which might
# be counted towards the query limit m, or just inefficient.
query_cache = {}

def cached_query(i, j):
    """ 
    A wrapper function for `query_judge` that utilizes memoization (caching).
    Checks if the result for query parameters (i, j) is already stored in `query_cache`.
    If yes, it returns the cached result directly.
    If not, it calls `query_judge(i, j)` to get the result from the judge, stores 
    this result in `query_cache` associated with the key (i, j), and then returns the result.
    """
    # Check if the result for this query (i, j) is already cached
    if (i, j) in query_cache:
        return query_cache[(i, j)]
    
    # If not cached, perform the query using query_judge
    result = query_judge(i, j)
    # Store the result in the cache
    query_cache[(i, j)] = result
    # Return the obtained result
    return result

def recursive_sum(node_i, node_j, target_L, target_R):
    """
    Recursively computes the sum of hidden array elements A_k for k in the target range [target_L, target_R].
    This function employs the standard algorithm for range sum queries on a segment tree.
    The segment tree structure is implicitly defined over the indices [0, 2^N - 1].
    Nodes in this conceptual tree correspond to dyadic intervals [2^i * j, 2^i * (j+1) - 1].
    
    Args:
        node_i: The level parameter 'i' for the query associated with the current node. 
                This determines the length of the interval covered by this node (2^node_i).
                The root is at level N, leaves are at level 0.
        node_j: The index parameter 'j' for the query associated with the current node.
                It specifies which interval of length 2^node_i this node covers.
                The interval starts at index 2^node_i * node_j.
        target_L: The starting index (inclusive) of the target range [L, R] for which we want the sum.
        target_R: The ending index (inclusive) of the target range [L, R].

    Returns:
        The sum (modulo 100) of elements A_k for indices k that are both within the interval 
        covered by the current node and within the target range [target_L, target_R].
    """
    
    # Calculate the interval [l, r] covered by the current node represented by (node_i, node_j)
    interval_len = 1 << node_i # Computes 2**node_i efficiently
    l = interval_len * node_j
    r = interval_len * (node_j + 1) - 1

    # Case 1: The node's interval [l, r] is completely outside the target range [target_L, target_R].
    # This means there is no overlap between the node's interval and the target range.
    # The contribution of this node (and its subtree) to the total sum is 0.
    if r < target_L or l > target_R:
        return 0

    # Case 2: The node's interval [l, r] is completely inside the target range [target_L, target_R].
    # This means the entire interval covered by this node is part of the range we need to sum up.
    # This node corresponds to a queryable dyadic interval [l, r].
    # We can directly ask the judge for the sum of this interval using the cached query function.
    if target_L <= l and r <= target_R:
        return cached_query(node_i, node_j)

    # Case 3: The node's interval partially overlaps with the target range [target_L, target_R].
    # This means some part of the node's interval is inside the target range, and some part is outside.
    # We cannot directly query this node. Instead, we must recursively call this function
    # on its children to compute the sum for the relevant parts.
    # This case only happens if the node is not a leaf (i.e., node_i > 0).
    # Leaf nodes (node_i = 0) represent single-element intervals [k, k]. A leaf node is always
    # either fully inside the target range (Case 2) or fully outside (Case 1). It cannot partially
    # overlap in a way that requires further decomposition.
    
    # Perform a defensive check for the leaf node case.
    if node_i == 0:
         # This code path should theoretically be unreachable if the logic for Case 1 and Case 2 is correct.
         # A leaf node [k, k] satisfies either k < target_L or k > target_R (Case 1), 
         # or target_L <= k <= target_R (Case 2).
         # If this path is reached, it might indicate a logical flaw or an unexpected edge case.
         # For robustness, return 0. Printing an error message might be useful for debugging.
         # print(f"Error: Leaf node ({node_i}, {node_j}) reached partial overlap case unexpectedly.", file=sys.stderr)
         return 0 
             
    # Recursively call the function for the left and right children of the current node.
    # The left child corresponds to query parameters (node_i - 1, 2 * node_j).
    # The right child corresponds to query parameters (node_i - 1, 2 * node_j + 1).
    sum_left = recursive_sum(node_i - 1, 2 * node_j, target_L, target_R)
    sum_right = recursive_sum(node_i - 1, 2 * node_j + 1, target_L, target_R)
    
    # The sum for the overlapping portion covered by the current node is the sum
    # of the results from its children. Remember to apply modulo 100.
    return (sum_left + sum_right) % 100

# Main execution block: runs when the script is executed.
if __name__ == '__main__':
    # Read the input integers N, L, and R from the first line of standard input.
    N, L, R = map(int, sys.stdin.readline().split())

    # Initiate the recursive calculation process to find the sum for the range [L, R].
    # The recursion starts from the root node of the conceptual segment tree.
    # The root node corresponds to query parameters (N, 0) and covers the full index range [0, 2^N - 1].
    total_sum = recursive_sum(N, 0, L, R)

    # After the recursive function computes the total sum (modulo 100),
    # print the final answer to standard output in the required format "! S".
    # Ensure the output is flushed to be immediately available to the judge.
    print(f"! {total_sum}", flush=True)