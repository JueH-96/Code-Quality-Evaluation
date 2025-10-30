# YOUR CODE HERE
import sys

# Increase recursion depth for deep DP calls; Python's default is often too low for competitive programming
# Set a higher limit suitable for N=300, potentially needing O(N^2) states depth in worst case? Better safe.
sys.setrecursionlimit(4000) 

MOD = 998244353

# Precompute Range Minimum Query indices using O(N log N) sparse table
# This allows finding the index of the minimum Y value in any range [L, R] in O(1) time after setup.
def build_sparse_table(arr):
    """Builds a sparse table for Range Minimum Query (RMQ) on the input array `arr`.
    Returns a query function `query_min_idx(L, R)` that gives the index of the minimum element
    in the subarray `arr[L...R]` in O(1) time.
    """
    N = len(arr)
    if N == 0: # Handle empty array case
         # Return a function that handles invalid queries gracefully
         return lambda L, R: -1 

    LOGN = N.bit_length() # Calculate max needed power of 2, effectively ceil(log2(N))
    # st_idx[j][i] stores the index of the minimum element in range [i, i + 2^j - 1]
    st_idx = [[0] * N for _ in range(LOGN)]
    
    # Initialize base level j=0: minimum in range [i, i] is just element at index i
    for i in range(N):
        st_idx[0][i] = i
    
    # Build sparse table levels iteratively
    for j in range(1, LOGN):
        # Iterate through starting positions i for ranges of length 2^j
        for i in range(N - (1 << j) + 1):
            # The range [i, i + 2^j - 1] is composed of two sub-ranges of length 2^(j-1):
            # [i, i + 2^(j-1) - 1] and [i + 2^(j-1), i + 2^j - 1]
            idx1 = st_idx[j-1][i]
            idx2 = st_idx[j-1][i + (1 << (j-1))]
            # Compare elements at indices found for sub-ranges to find the index of the minimum for the current range
            if arr[idx1] <= arr[idx2]: # Use <= to prefer left index in case of ties
                st_idx[j][i] = idx1
            else:
                st_idx[j][i] = idx2
    
    # Precompute log base 2 values to quickly find the appropriate level k for queries
    # log_table[x] = floor(log2(x))
    log_table = [0] * (N + 1)
    for i in range(2, N + 1):
        log_table[i] = log_table[i // 2] + 1

    # Define the query function using the precomputed tables
    def query_min_idx(L, R):
         """Queries the sparse table for the index of the minimum element in arr[L...R]."""
         # Check for invalid range definitions
         if L < 0 or R >= N or L > R: return -1 
         
         length = R - L + 1
         k = log_table[length] # Determine the largest power of 2 less than or equal to length
         # Find minimum in two potentially overlapping ranges that cover [L, R]:
         # [L, L + 2^k - 1] and [R - 2^k + 1, R]
         idx1 = st_idx[k][L]
         idx2 = st_idx[k][R - (1 << k) + 1]
         
         # Return index of the overall minimum by comparing the minima of the two ranges
         if arr[idx1] <= arr[idx2]: # Use <= to prefer left index in case of ties
             return idx1
         else:
             return idx2
             
    return query_min_idx

# Dictionary for memoization of DP states. Key is tuple (l, r), value is computed result.
dp = {} 

# DP function definition using recursion with memoization
# dp state (l, r) represents computation for subarray P[l..r]
# The recurrence relation used here is based on patterns observed in similar competitive programming problems,
# often related to counting structures (like trees, paths, partitions) on intervals determined by minima/maxima.
# It was found during the thought process that this specific recurrence yields incorrect results on samples for this problem.
# It is presented here as it was the most structurally relevant pattern explored. The actual solution might require a different DP formulation.
def get_dp(l, r, P, query_min_idx):
    """Recursive DP function with memoization."""
    state = (l, r)
    # Base case: empty interval. Represents 1 way (context-dependent meaning, often identity element for multiplication).
    if l > r:
        return 1
    # Return memoized result if this state has already been computed
    if state in dp:
        return dp[state]
    
    # Find index k of the minimum Y value in the current interval [l, r] using the RMQ structure
    k = query_min_idx(l, r)
    
    # Check if k is valid (should always be for non-empty valid range l <= r)
    if k == -1: 
         # This case handles potential errors or empty initial array passed to build_sparse_table
         # For valid L,R with N>0, k should be in [L, R].
         return 1 # Default return for safety, though ideally should not be reached in valid execution path.

    # Calculate left summation part of the recurrence
    # Sum over p from l to k
    left_sum = 0
    for p in range(l, k + 1):
        # Recursive calls for subproblems [l, p-1] and [p, k-1]
        term = (get_dp(l, p - 1, P, query_min_idx) * get_dp(p, k - 1, P, query_min_idx)) % MOD
        left_sum = (left_sum + term) % MOD
    
    # Calculate right summation part of the recurrence
    # Sum over q from k to r
    right_sum = 0
    for q in range(k, r + 1):
         # Recursive calls for subproblems [k+1, q] and [q+1, r]
         term = (get_dp(k + 1, q, P, query_min_idx) * get_dp(q + 1, r, P, query_min_idx)) % MOD
         right_sum = (right_sum + term) % MOD

    # Combine left and right parts using multiplication, take modulo
    res = (left_sum * right_sum) % MOD
    # Store result in memoization table before returning
    dp[state] = res
    return res

def solve():
    """Main function to read input, process, and print the result."""
    N = int(sys.stdin.readline())
    
    # Handle trivial case N=0
    if N == 0:
        print(0)
        return

    points = []
    for i in range(N):
        # Read points and store original index (potentially useful, though unused in this DP)
        points.append(list(map(int, sys.stdin.readline().split())) + [i]) 

    # Sort points based on X coordinate. This transforms the problem into operating on a permutation P 
    # representing Y coordinates corresponding to sorted X coordinates.
    points.sort()

    # Extract Y coordinates into array P based on sorted X order
    P = [0] * N 
    for i in range(N):
       P[i] = points[i][1] 
    
    # Build the sparse table for Range Minimum Query on Y coordinates (P array)
    query_min_idx = build_sparse_table(P)
    
    # Clear DP memoization table for safety (e.g. if running multiple tests in one execution)
    dp.clear() 

    # Compute the final answer using the DP function for the full range [0, N-1]
    # Note: This DP formulation was found to be incorrect for the problem based on sample testing.
    # The correct approach might require a different DP state or transition logic.
    print(get_dp(0, N - 1, P, query_min_idx))

# Execute the main solve function when the script is run
solve()