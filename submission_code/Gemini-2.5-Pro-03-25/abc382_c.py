# YOUR CODE HERE
import sys

# Increase recursion depth limit for potentially deep segment tree traversals.
# While N=2e5 leads to depth ~18 which is usually within Python's default limit,
# setting it higher is a safety measure in competitive programming.
try:
    # Set a large enough recursion depth limit. 
    # N <= 2*10^5, so 2*10^5 + 10 should be safe.
    sys.setrecursionlimit(200010) 
except Exception as e:
    # If setting recursion depth fails (e.g., due to OS limits), 
    # proceed with the default limit.
    pass 

def solve():
    # Read N (number of people) and M (number of sushi pieces)
    N, M = map(int, sys.stdin.readline().split())
    
    # Read gourmet levels A_1, ..., A_N for N people
    # Constraints state N, M >= 1.
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read deliciousness values B_1, ..., B_M for M sushi pieces
    B = list(map(int, sys.stdin.readline().split()))

    # Calculate the required size for the segment tree array. 
    # 4*N is a safe upper bound for a binary tree based segment tree implementation.
    st_size = 4 * N 
    
    # Set a value representing infinity or 'not applicable' for segment tree nodes.
    # It should be larger than any possible gourmet level A_i. Max A_i is 2*10^5.
    MAX_VAL = 2 * 10**5 + 1 
    # Initialize the segment tree array with this large value.
    st = [MAX_VAL] * st_size
    
    # Define a constant representing 'not found' for the query result.
    # It must be an index larger than any possible person index N.
    INF = N + 1 

    # Build function for the segment tree.
    # It computes the minimum gourmet level in the range [L, R] and stores it at node k.
    # k: current node index in the segment tree array (1-based indexing for nodes)
    # L, R: range of people indices covered by node k (1-based indexing for people)
    def build(k, L, R):
        # Base case: If the range contains a single person L
        if L == R:
            # Store the gourmet level of person L (A[L-1] using 0-based list index)
            st[k] = A[L-1] 
            return
        
        # Recursive step: Divide the range [L, R] into two halves [L, M] and [M+1, R]
        M = (L + R) // 2
        # Build the left subtree
        build(2*k, L, M)
        # Build the right subtree
        build(2*k + 1, M + 1, R)
        # The value at the current node k is the minimum of its children's values
        st[k] = min(st[2*k], st[2*k + 1])

    # Query function for the segment tree.
    # It finds the minimum index i in the range [L, R] such that A[i-1] <= targetB.
    # k: current node index
    # L, R: range covered by current node k
    # targetB: the deliciousness value of the sushi being queried
    # Returns the minimum person index (1-based) or INF if no such person is found.
    def query(k, L, R, targetB):
        # Pruning optimization: If the minimum gourmet level in the range [L, R] (st[k])
        # is already greater than the sushi's deliciousness (targetB), then no person
        # in this range can eat the sushi. Return INF immediately.
        if st[k] > targetB:
            return INF
        
        # Base case: If we have reached a leaf node representing a single person L
        if L == R:
            # Since we passed the `st[k] > targetB` check above, we know st[k] <= targetB.
            # This means person L (whose gourmet level is st[k]) can eat the sushi.
            # As this is a leaf node, return the index L.
            return L 

        # Recursive step: Search in subtrees. Calculate midpoint M.
        M = (L + R) // 2
        
        # Prioritize searching the left child (range [L, M]) first.
        # This is because we want the minimum index, and indices in the left subtree
        # are always smaller than indices in the right subtree.
        left_res = query(2*k, L, M, targetB)
        
        # If a suitable person is found in the left subtree (result is not INF),
        # return this result immediately. It's guaranteed to be the minimum index.
        if left_res != INF:
             return left_res
        
        # If no suitable person was found in the left subtree (left_res is INF),
        # then search the right subtree (range [M+1, R]).
        # The result from the right subtree (which could be an index or INF) is returned.
        right_res = query(2*k + 1, M + 1, R, targetB)
        return right_res

    # Build the segment tree using the provided gourmet levels A.
    # Start the build process from the root node (index 1) covering range [1, N].
    build(1, 1, N)

    # Process each sushi query.
    results = [] # List to store the results for each sushi
    for j in range(M):
        # Get the deliciousness of the current sushi B[j]
        targetB = B[j]
        # Perform the query to find the index of the first person who eats this sushi.
        # Start query from the root node (index 1) covering the full range [1, N].
        result_idx = query(1, 1, N, targetB)
        
        # Check the query result.
        if result_idx == INF:
            # If INF is returned, no person ate the sushi. Append "-1".
            results.append("-1")
        else:
            # Otherwise, person with index `result_idx` ate the sushi. Append their index as a string.
            results.append(str(result_idx))

    # Print all collected results, each on a new line.
    print('
'.join(results))

# Execute the solve function when the script runs.
solve()