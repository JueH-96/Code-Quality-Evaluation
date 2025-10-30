import math
from typing import List

# Use a large number for infinity
INF = float('inf')

class SegTree:
    """ 
    Basic Segment Tree implementation for Range Minimum Query (RMQ).
    It supports building the tree from an array in O(N) time and querying the minimum
    value in a given range [L, R] in O(log N) time.
    """
    def __init__(self, arr):
        self.n = len(arr)
        # Initialize tree with INF. Size 4*N is sufficient for binary tree structure.
        self.tree = [INF] * (4 * self.n) 
        self.arr = arr
        if self.n > 0:
            self.build(0, 0, self.n - 1)

    def build(self, node, start, end):
        """ Recursively builds the segment tree """
        if start == end:
            # Leaf node stores the value from the original array
            self.tree[node] = self.arr[start]
        else:
            mid = (start + end) // 2
            # Build left and right children
            self.build(2 * node + 1, start, mid)
            self.build(2 * node + 2, mid + 1, end)
            # Internal node stores the minimum of its children
            self.tree[node] = min(self.tree[2 * node + 1], self.tree[2 * node + 2])

    def query(self, node, start, end, L, R):
        """ Recursive helper function for range query """
        # If the query range is completely outside the node's range, return INF
        if R < start or end < L:
            return INF
        # If the node's range is completely contained within the query range, return the node's value
        if L <= start and end <= R:
            return self.tree[node]
        # If the query range partially overlaps the node's range, query children
        mid = (start + end) // 2
        p1 = self.query(2 * node + 1, start, mid, L, R)
        p2 = self.query(2 * node + 2, mid + 1, end, L, R)
        # Return the minimum of the results from children
        return min(p1, p2)

    def query_range(self, L, R):
        """ Public method to query the minimum value in range [L, R] """
        # Handle invalid or empty ranges
        if self.n == 0 or L > R: return INF
        # Clamp query indices to valid array bounds [0, n-1]
        L = max(0, L)
        R = min(self.n - 1, R)
        # Check again if adjusted range is valid
        if L > R: return INF
        # Perform the query using the recursive helper function
        return self.query(0, 0, self.n - 1, L, R)

class Solution:
    """
    Solves the problem using Dynamic Programming with optimizations.
    DP state dp[k][i] = minimum value sum for partitioning nums[i:] into m-k subarrays.
    Uses Sparse Table for O(1) Range AND Query. Precomputation O(N log N).
    Uses Segment Tree for O(log N) Range Minimum Query on computed costs. Build O(N), Query O(log N).
    Overall Time Complexity: O(N log N + M * N log N) = O(M N log N)
    Space Complexity: O(N log N + MN) for sparse table and DP table. O(N) for segment tree. Total O(N log N + MN).
    """
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        
        # Handle edge case: empty nums array. If m=0, cost is 0. If m>0, impossible.
        if n == 0: 
             return 0 if m == 0 else -1

        # Precompute floor(log2(x)) values to find Sparse Table dimensions and assist queries
        LOG2 = [0] * (n + 1)
        for i in range(2, n + 1):
            LOG2[i] = LOG2[i // 2] + 1
        # Maximum power of 2 needed for sparse table (max k such that 2^k <= n)
        MAX_K = LOG2[n] 

        # Build sparse table `st` for range AND queries. st[i][k] stores AND of nums[i..i + 2^k - 1]
        st = [[0] * (MAX_K + 1) for _ in range(n)]
        for i in range(n):
            st[i][0] = nums[i]
        for k in range(1, MAX_K + 1):
            pow2k_minus_1 = 1 << (k - 1) # 2^(k-1)
            pow2k = 1 << k # 2^k
            # Iterate through possible start indices i for intervals of length 2^k
            for i in range(n - pow2k + 1):
                # Combine results from two overlapping intervals of length 2^(k-1) from level k-1
                st[i][k] = st[i][k - 1] & st[i + pow2k_minus_1][k - 1]

        # Helper function for O(1) range AND query using the precomputed sparse table
        def query_and(L, R):
            # Handle invalid range: return -1 (all bits 1, identity for AND)
            if L > R: return -1 
            # Ensure indices are valid before calculating k. Check bounds to prevent errors.
            if L < 0 or R >= n or L >= n or R < 0: return -1 # Defensive check, should not happen with correct logic
            
            length = R - L + 1
            if length <= 0: return -1 # Defensive check
            
            # Determine the largest power of 2 (k) such that 2^k <= length
            k = LOG2[length]
            pow2k = 1 << k
            # Combine results from two overlapping intervals covering [L, R]: [L, L+2^k-1] and [R-2^k+1, R]
            return st[L][k] & st[R - pow2k + 1][k]

        # Initialize DP table. dp[k][i] stores the minimum cost for the suffix nums[i:] using m-k subarrays
        # Dimensions: (m+1) rows (for k=0..m), (n+1) columns (for i=0..n)
        dp = [[INF] * (n + 1) for _ in range(m + 1)]
        # Base case: partitioning an empty suffix (nums[n:]) into 0 subarrays (k=m) costs 0.
        dp[m][n] = 0

        # Iterate through DP states: layers k from m-1 down to 0
        for k in range(m - 1, -1, -1):
            # Prepare the cost array C_k for the current layer k.
            # C_k[j] = nums[j] + dp[k+1][j+1] represents the minimum cost if the (k+1)-th subarray ends at index j.
            C_k = [INF] * n
            for j in range(n):
                # Get the minimum cost from the next state (k+1) starting at index j+1
                next_dp_val = dp[k + 1][j + 1] 
                if next_dp_val != INF:
                     # If the next state is reachable (cost is not INF), calculate the total cost
                     # The value of the current subarray ending at j is nums[j].
                     C_k[j] = nums[j] + next_dp_val
            
            # Build a Segment Tree on the cost array C_k for efficient RMQ
            seg_tree = SegTree(C_k)

            # Compute dp[k][i] for each possible start index i of the (k+1)-th subarray
            # Iterate i from n-1 down to 0 (to ensure required subproblems dp[k+1][...] are computed)
            for i in range(n - 1, -1, -1):
                target_V = andValues[k] # Target AND value for the (k+1)-th subarray
                
                # Find the range [p, q] such that for any j in [p, q], query_and(i, j) == target_V.
                # Use binary search twice to find the boundaries p and q efficiently.

                # Find the smallest index p >= i such that query_and(i, p) == target_V
                # First binary search finds the smallest index p_candidate >= i where query_and(i, p_candidate) <= target_V
                low = i
                high = n - 1
                p = -1 # Initialize p to indicate not found
                while low <= high:
                    mid = (low + high) // 2
                    val = query_and(i, mid)
                    if val <= target_V:
                        # If val matches target_V, this `mid` is a potential candidate for `p`.
                        # We store it and continue searching left for possibly smaller index.
                        if val == target_V:
                            p = mid
                        # Search in the left half [low, mid-1] regardless of match to find the *smallest* p
                        high = mid - 1 
                    else: # val > target_V
                        # Current AND value is too high, need to include more elements (extend to right)
                        low = mid + 1
                
                # If p remains -1, or if the final p found does not yield exactly target_V, then no valid subarray starts at i.
                if p == -1: # This check implies query_and(i,p) != target_V was already implicitly handled by the loop condition update logic
                    continue # Skip to next i, dp[k][i] remains INF

                # Find the largest index q >= p such that query_and(i, q) == target_V
                # Second binary search finds the largest index q >= p where query_and(i, q) >= target_V (implies == target_V)
                low = p
                high = n - 1
                q = p # Initialize q to p (at least p works)
                while low <= high:
                    mid = (low + high) // 2
                    val = query_and(i, mid)
                    if val == target_V: 
                        # Found a candidate `q`, potentially update `q` and search right for larger index.
                        q = mid
                        low = mid + 1
                    elif val < target_V: 
                        # AND value dropped below target_V, need to shorten the interval (search left)
                        high = mid - 1
                    else: # val > target_V
                        # This case should be impossible given query_and is non-increasing and query_and(i, p) == target_V
                        # If it occurs, something is wrong. Treat as too high, search left.
                        high = mid - 1 

                # After binary searches, we have the range [p, q] of valid end indices j for the subarray starting at i.
                # Query the segment tree for the minimum cost in C_k over indices [p, q].
                min_cost = seg_tree.query_range(p, q)
                
                # If a valid path was found (min_cost is not INF), update dp[k][i]
                if min_cost != INF:
                   dp[k][i] = min_cost
                # Otherwise, dp[k][i] remains INF, indicating impossibility from this state.

        # The final answer is the minimum cost to partition the entire array nums[0..n-1] into m subarrays.
        result = dp[0][0]
        # Return the result if finite, otherwise return -1 indicating impossibility.
        return int(result) if result != INF else -1