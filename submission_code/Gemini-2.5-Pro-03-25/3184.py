import math
from typing import List

# Define a Segment Tree class that supports Range Maximum Query and Point Update (taking maximum).
# The tree nodes are 1-indexed. The data range (ranks) is 0-indexed [0, m-1].
class SegmentTree:
    def __init__(self, size):
        """
        Initializes the segment tree.
        'size' is the number of elements in the base array (number of ranks 'm').
        The tree itself will have roughly 4*size nodes.
        Initialize tree nodes with negative infinity, as it's the identity for max operation
        and represents 'no element found' or 'no valid path'.
        """
        self.size = size
        # Use 4*size array for safety, standard practice for segment tree implementation.
        self.tree = [-float('inf')] * (4 * size) 

    def update(self, node: int, start: int, end: int, idx: int, val: int):
        """
        Updates the segment tree. Sets the value at index 'idx' to max(current_value, val).
        'node' is the current tree node index (1-based).
        'start', 'end' define the range covered by the current node.
        'idx' is the index in the original data range (0-based rank) to update.
        'val' is the new value to consider for updating.
        """
        # Base case: If we reach the leaf node corresponding to index idx
        if start == end:
            # Update the leaf node value by taking the maximum with existing value.
            # This is needed because multiple original indices `i` might map to the same rank `idx`.
            # We want the segment tree to store the maximum dp value achieved for that rank.
            self.tree[node] = max(self.tree[node], val)
            return

        mid = (start + end) // 2
        # Recurse down to the correct child based on index 'idx'
        if idx <= mid:
            self.update(2 * node, start, mid, idx, val)
        else:
            self.update(2 * node + 1, mid + 1, end, idx, val)
        
        # After updating child(ren), update the current node's value.
        # Internal node stores the maximum of its children.
        self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, node: int, start: int, end: int, L: int, R: int) -> int:
        """
        Queries the segment tree for the maximum value in the range [L, R].
        'node' is the current tree node index (1-based).
        'start', 'end' define the range covered by the current node.
        'L', 'R' define the query range (0-based ranks).
        Returns the maximum value in the range [L, R].
        """
        # If the query range is completely outside the node's range
        if R < start or end < L:
            # Return negative infinity indicating no overlap or element found in query range by this path.
            return -float('inf') 
        
        # If the node's range is completely within the query range
        if L <= start and end <= R:
            return self.tree[node]

        # If the node's range partially overlaps with the query range
        mid = (start + end) // 2
        # Recursively query left and right children
        left_query = self.query(2 * node, start, mid, L, R)
        right_query = self.query(2 * node + 1, mid + 1, end, L, R)
        # Return the maximum of the results from children
        return max(left_query, right_query)

class Solution:
    """
    Finds the maximum sum of a balanced subsequence of nums.
    The condition for a subsequence with indices i_0 < i_1 < ... < i_k-1 to be balanced is:
    nums[i_j] - nums[i_j-1] >= i_j - i_j-1 for all 1 <= j < k.
    This is equivalent to nums[i_j] - i_j >= nums[i_j-1] - i_j-1.
    Let b[i] = nums[i] - i. The condition becomes b[i_j] >= b[i_j-1].
    The problem is transformed to finding a subsequence with indices i_0 < ... < i_k-1
    such that b[i_0] <= b[i_1] <= ... <= b[i_k-1] and the sum nums[i_0] + ... + nums[i_k-1] is maximized.

    This problem structure is similar to finding the Longest Increasing Subsequence (LIS),
    but instead of maximizing length, we maximize the sum. We use dynamic programming optimized with a Segment Tree.
    Define dp[i] as the maximum sum of a balanced subsequence ending at index i.
    The recurrence relation becomes:
    dp[i] = nums[i] + max(0, M) where M = max({dp[j] | j < i and b[j] <= b[i]}).
    During the thought process, the DP formulation was refined to:
    dp[i] = max(nums[i], nums[i] + M) where M = max({dp value associated with rank k' | k' <= rank(b[i])}).
    If no such previous element exists (set is empty), M = -inf. The logic handles this.

    The computation of M is efficiently done using a Segment Tree data structure.
    The Segment Tree operates on the ranks of the distinct values of b[i].
    Coordinate compression is used to map potentially large or negative b[i] values to ranks [0, m-1].
    """
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Step 1: Transform the array nums into array b where b[i] = nums[i] - i.
        b = [nums[i] - i for i in range(n)]
        
        # Step 2: Coordinate Compression on array b.
        # Get unique sorted values of b.
        distinct_b = sorted(list(set(b)))
        # Create a mapping from each unique value to its rank (0-based index).
        val_to_rank = {val: k for k, val in enumerate(distinct_b)}
        m = len(distinct_b) # The number of unique values, which is the size of our effective domain for DP states.
        
        # Step 3: Initialize Segment Tree.
        # The segment tree will store the maximum dp sums achieved for each rank.
        st = SegmentTree(m)
        
        # Initialize the overall maximum sum found so far.
        max_overall_sum = -float('inf')

        # Step 4: Iterate through the input array and compute DP values.
        for i in range(n):
            # Get the rank of the current element's b value.
            rank = val_to_rank[b[i]]
            
            # Query the Segment Tree for the maximum DP value among elements processed so far
            # with rank less than or equal to the current rank. This gives M.
            # Query range is [0, rank]. Tree uses 1-based node index, range [0, m-1].
            M = st.query(1, 0, m - 1, 0, rank)
            
            # Calculate the maximum balanced subsequence sum ending at index i.
            # Initialize with nums[i] (case: subsequence of length 1 starting at i).
            current_dp = nums[i]
            # If M is not -infinity, it means there exists at least one previous element j < i
            # such that b[j] <= b[i]. We can potentially extend the best such subsequence.
            if M > -float('inf'):
                 # The sum would be nums[i] + M. We take the maximum of starting anew vs extending.
                 current_dp = max(current_dp, nums[i] + M)

            # Step 5: Update the Segment Tree.
            # Store the computed `current_dp` value associated with the current element's rank.
            # The update function handles taking the maximum if a value already exists for this rank.
            st.update(1, 0, m - 1, rank, current_dp)
            
            # Step 6: Update the overall maximum sum found.
            # The maximum sum can end at any index i.
            max_overall_sum = max(max_overall_sum, current_dp)
            
        # The final result is the maximum sum found across all possible ending indices.
        return max_overall_sum