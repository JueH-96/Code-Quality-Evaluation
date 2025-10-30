from typing import List
# import bisect # Not strictly needed if using the map

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)

        # Step 1 & 2: Calculate v_i values and get sorted unique values
        # Define v_i = nums[i] - i. The balanced subsequence condition
        # nums[i_j] - nums[i_j-1] >= i_j - i_j-1 is equivalent to
        # nums[i_j] - i_j >= nums[i_j-1] - i_j-1, which is v_{i_j} >= v_{i_j-1}.
        # So, a subsequence i_0 < i_1 < ... < i_k-1 is balanced if v_{i_0} <= v_{i_1} <= ... <= v_{i_{k-1}}.
        v_values = sorted(list(set(nums[i] - i for i in range(n))))
        
        # Create a map from v_value to its compressed index (rank).
        # This allows us to use a segment tree on a contiguous range of indices [0, num_unique_v - 1].
        v_map = {val: i for i, val in enumerate(v_values)}
        num_unique_v = len(v_values)

        # Step 3: Initialize Segment Tree
        # The segment tree will store the maximum dp value for each compressed v_value index.
        # The indices of the segment tree will correspond to the compressed v_values [0, num_unique_v - 1].
        # A size of 4 * num_unique_v is a safe upper bound for the segment tree array size.
        tree_size = 4 * num_unique_v
        # Initialize tree nodes with -float('inf') as we are finding maximums.
        # -float('inf') is the identity element for the max operation.
        tree = [-float('inf')] * tree_size

        # Segment Tree Functions
        def update(idx, tree_L, tree_R, target_k, val):
            """
            Update the maximum value associated with the compressed index target_k.
            The update propagates from the leaf node up to the root.
            idx: current node index in the tree array
            tree_L, tree_R: range of compressed indices covered by the current node [inclusive]
            target_k: the compressed index (leaf) to update
            val: the new value to potentially update with (max of current and new value at the leaf)
            """
            # Base case: reached the leaf node corresponding to target_k
            if tree_L == tree_R:
                # Update the value at this leaf node. We take max because multiple
                # original indices might map to the same v_value, and we want the
                # maximum DP sum ending at any of those indices.
                tree[idx] = max(tree[idx], val)
                return

            # Recursive step: determine which child node covers target_k
            mid = (tree_L + tree_R) // 2
            if target_k <= mid:
                # target_k is in the left child's range
                update(2 * idx + 1, tree_L, mid, target_k, val)
            else:
                # target_k is in the right child's range
                update(2 * idx + 2, mid + 1, tree_R, target_k, val)

            # After updating the child, update the current node value
            # The value at an internal node is the maximum of its children
            tree[idx] = max(tree[2 * idx + 1], tree[2 * idx + 2])

        def query(idx, tree_L, tree_R, query_L, query_R):
            """
            Query the maximum value in the compressed index range [query_L, query_R].
            idx: current node index in the tree array
            tree_L, tree_R: range of compressed indices covered by the current node [inclusive]
            query_L, query_R: the query range of compressed indices [inclusive]
            """
            # Case 1: Current node range is completely outside the query range
            if query_L > tree_R or query_R < tree_L:
                return -float('inf') # Return identity element for max operation

            # Case 2: Current node range is completely inside the query range
            if query_L <= tree_L and tree_R <= query_R:
                return tree[idx]

            # Case 3: Partial overlap, recurse on children
            mid = (tree_L + tree_R) // 2
            left_max = query(2 * idx + 1, tree_L, mid, query_L, query_R)
            right_max = query(2 * idx + 2, mid + 1, tree_R, query_L, query_R)

            # The maximum in the current node's range overlapping with the query range
            # is the maximum of the results from its children
            return max(left_max, right_max)

        # Step 4 & 5: Iterate through nums and calculate DP
        # dp[i] = maximum sum of a balanced subsequence ending at index i.
        # dp[i] = nums[i] + max({0} U {dp[j] | j < i and v_j <= v_i})
        # The max({0} U ...) handles the case where nums[i] starts a new subsequence.
        # The term {dp[j] | j < i and v_j <= v_i} is the maximum dp value among
        # previous indices j that satisfy the balanced condition with i.
        
        # max_overall_sum will store the maximum dp[i] found so far across all i.
        max_overall_sum = -float('inf')

        for i in range(n):
            # Calculate v_i and its compressed index
            v_i = nums[i] - i
            compressed_v_i = v_map[v_i]

            # Query the maximum dp value among all balanced subsequences ending
            # at a previous index j < i, such that v_j <= v_i.
            # This corresponds to querying the segment tree for the maximum value
            # in the compressed index range [0, compressed_v_i].
            # The query function handles the range [query_L, query_R] inclusive.
            # If no such j exists or their dp values are all -inf, query returns -inf.
            max_prev_dp = query(0, 0, num_unique_v - 1, 0, compressed_v_i)

            # Calculate dp[i]: max sum of a balanced subsequence ending at index i.
            # nums[i] is the current element. We add the maximum sum of a valid
            # preceding balanced subsequence. The `max(0, max_prev_dp)` term
            # correctly implements the max({0} U ...) part. If max_prev_dp is -inf
            # or negative, max(0, max_prev_dp) is 0, meaning we start a new
            # subsequence with nums[i].
            current_dp = nums[i] + max(0, max_prev_dp)

            # Update the segment tree at the compressed index corresponding to v_i
            # with the newly calculated current_dp. This makes current_dp available
            # for future queries when considering indices k > i.
            update(0, 0, num_unique_v - 1, compressed_v_i, current_dp)

            # Update the overall maximum balanced subsequence sum found so far
            max_overall_sum = max(max_overall_sum, current_dp)

        # After iterating through all elements, max_overall_sum holds the maximum
        # dp[i] value over all i, which is the maximum sum of any balanced subsequence
        # ending at any index i. This is the maximum balanced subsequence sum overall.
        # Since current_dp >= nums[i], max_overall_sum will be at least max(nums),
        # correctly covering the case of single-element balanced subsequences.
        return max_overall_sum