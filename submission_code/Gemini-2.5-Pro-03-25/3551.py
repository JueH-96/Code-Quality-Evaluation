import math
from typing import List

# Helper function to precompute base-2 logarithms up to N
# This is used for sparse table construction and queries.
def _compute_log_table(N):
    """ Precomputes base-2 logarithms for integers up to N. """
    LOG2 = [0] * (N + 1)
    # LOG2[i] will store floor(log2(i))
    for i in range(2, N + 1):
        LOG2[i] = LOG2[i // 2] + 1
    return LOG2

class Solution:
  """
  Solves the Maximum Subarray XOR Score problem using dynamic programming and sparse tables.

  The problem asks for the maximum XOR score among all subarrays within a given range [l, r] for several queries.
  The XOR score of an array `a` is defined by a specific recursive process involving XORing adjacent elements and removing the last element.
  This process leads to a final score which can be computed efficiently.

  The solution strategy is as follows:
  1. Precompute the XOR score `scores[i][j]` for all possible subarrays `nums[i..j]`. This is done using a dynamic programming approach based on the recurrence relation S(i, j) = S(i, j-1) ^ S(i+1, j), with base cases S(i, i) = nums[i].
  2. Precompute `MaxRow[i][r]`, which stores the maximum XOR score among all subarrays starting at index `i` and ending at any index `k` such that `i <= k <= r`. This is computed as `MaxRow[i][r] = max_{k=i}^r scores[i][k]`.
  3. Build Sparse Tables for each column `r` of the `MaxRow` table. Each sparse table allows for efficient O(1) Range Maximum Queries (RMQ) on the corresponding column.
  4. Answer each query `[l, r]` by performing an RMQ on the sparse table for column `r` over the range `[l, r]`. The result `max_{i=l}^r MaxRow[i][r]` gives the maximum XOR score among all subarrays `nums[i..j]` such that `l <= i <= j <= r`.

  Time Complexity: O(N^2 log N + Q), where N is the length of `nums` and Q is the number of queries. The dominant part is the construction of N sparse tables, each taking O(N log N). Query answering takes O(Q) total time.
  Space Complexity: O(N^2 log N), primarily due to storing the N sparse tables. Each sparse table takes O(N log N) space.
  """
  def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
    """
    Calculates the maximum XOR score for subarrays within specified query ranges.

    Args:
        nums: The input list of non-negative integers.
        queries: A list of queries, where each query is represented as [l_i, r_i],
                 defining the range [l_i, r_i] in the nums array.

    Returns:
        A list of integers, where the i-th element is the maximum XOR score
        found within the range specified by the i-th query.
    """
    N = len(nums)
    
    # Constraints state 1 <= N <= 2000. Handle N=0 just in case, though not expected by constraints.
    if N == 0:
        return [0] * len(queries)

    # 1. Precompute XOR scores `scores[i][j]` for all subarrays `nums[i..j]`
    # Time: O(N^2), Space: O(N^2)
    scores = [[0] * N for _ in range(N)]
    for i in range(N):
        scores[i][i] = nums[i]  # Base case: subarray of length 1
    
    # Compute scores for subarrays of length 2 up to N
    for length in range(2, N + 1):
        for i in range(N - length + 1): # Start index `i`
            j = i + length - 1 # End index `j`
            # Apply the recurrence relation S(i, j) = S(i, j-1) ^ S(i+1, j)
            scores[i][j] = scores[i][j-1] ^ scores[i+1][j]

    # 2. Precompute `MaxRow[i][r] = max_{k=i}^r scores[i][k]`
    # MaxRow[i][r] stores the maximum score among subarrays starting at `i` and ending at or before `r`.
    # Time: O(N^2), Space: O(N^2)
    MaxRow = [[0] * N for _ in range(N)]
    for i in range(N):
        MaxRow[i][i] = scores[i][i] # Base case: r = i
        # Compute MaxRow[i][r] for r > i using the previously computed value MaxRow[i][r-1]
        for r in range(i + 1, N):
            MaxRow[i][r] = max(MaxRow[i][r-1], scores[i][r])

    # Precompute Log table using helper function for sparse table dimensions/queries
    # Time: O(N), Space: O(N)
    LOG2 = _compute_log_table(N)

    # 3. Build Sparse Tables for Range Maximum Query (RMQ) on columns of MaxRow
    # The goal is to quickly find max_{i=l}^r MaxRow[i][r] for any query [l, r].
    # Time: O(N^2 log N), Space: O(N^2 log N)
    K = LOG2[N] if N > 0 else 0 # Maximum power of 2 needed for sparse table height
    
    # sparse_tables_max[r][k][i] will store the maximum value in the range [i, i + 2^k - 1]
    # of the r-th column of the MaxRow table.
    sparse_tables_max = [[[0]*N for _ in range(K + 1)] for _ in range(N)]

    for r in range(N): # Iterate through each column index r
        st = sparse_tables_max[r] # Get the structure for the sparse table of column r
        
        # Initialize base level k=0 of the sparse table using values from MaxRow column r
        # MaxRow[i][r] is defined only for i <= r. Other entries should be 0.
        for i in range(r + 1): 
            st[0][i] = MaxRow[i][r]
        # Indices i > r already have st[0][i] = 0 by initialization.
        
        # Build higher levels k=1 to K
        for k in range(1, K + 1):
            # The loop calculates max for ranges of length 2^k.
            # It iterates up to the last possible start index `i` such that the range [i, i + 2^k - 1]
            # stays within the bounds [0, N-1]. Max `i` is N - (1 << k).
            for i in range(N - (1 << k) + 1):
                # Combine results from two half-ranges of level k-1
                st[k][i] = max(st[k-1][i], st[k-1][i + (1 << (k-1))])
            
    # 4. Answer queries using the precomputed sparse tables
    # Time: O(Q), Space: O(Q) for the results array
    ans = []
    for l, r in queries:
        # For a query [l, r], we need the maximum value in the range [l, r] of column r of MaxRow.
        query_len = r - l + 1
        
        # Check for invalid query length (should not happen per constraints l <= r)
        if query_len <= 0:
             ans.append(0) # Default answer for invalid range
             continue
        
        # Determine the largest power of 2, k, such that 2^k <= query_len
        k = LOG2[query_len]
        
        # Get the appropriate sparse table for the column index 'r'
        st_r = sparse_tables_max[r]
        
        # Perform the RMQ using the standard sparse table query formula:
        # max(st[k][L], st[k][R - 2^k + 1]) covers the entire range [L, R]
        result = max(st_r[k][l], st_r[k][r - (1 << k) + 1])
        ans.append(result)

    return ans