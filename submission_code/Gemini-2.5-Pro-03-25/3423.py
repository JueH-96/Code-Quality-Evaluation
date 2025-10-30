import sys
from typing import List

# Optional: Adjust recursion depth if necessary, although Python's default is usually sufficient
# sys.setrecursionlimit(2000) 

# Define a large negative integer to represent negative infinity.
# It must be smaller than any possible valid sum. Minimum possible sum is related to N * min(nums[i]),
# but since we take max(0, val) for included elements, intermediate sums are non-negative or NEG_INF.
# A large negative number like -10^18 is safe.
NEG_INF = -1 * (10**18) 

# Define the modulo value as required by the problem.
MOD = 10**9 + 7

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        """
        Calculates the sum of maximum subsequence sums for multiple queries.
        Each query updates an element in nums and then computes the maximum sum
        of a subsequence with no adjacent elements selected.

        Args:
            nums: The initial list of integers.
            queries: A list of queries, where each query is [pos_i, x_i] indicating
                     to set nums[pos_i] = x_i.

        Returns:
            The total sum of answers for all queries, modulo 10^9 + 7.
        """
        N = len(nums)
        
        # Handle edge case of empty input array, though constraints state N >= 1.
        if N == 0:
            return 0
        
        # Initialize the segment tree array. We use 1-based indexing for nodes.
        # Size 4*N is generally sufficient for a binary tree structure.
        # Each node stores a tuple (S00, S01, S10, S11) representing the max sums
        # under different boundary conditions (0=excluded, 1=included for Left/Right ends).
        tree = [(NEG_INF, NEG_INF, NEG_INF, NEG_INF)] * (4 * N) 

        # Helper function for safe addition involving NEG_INF.
        # If either operand is NEG_INF, the result is NEG_INF.
        def safe_add(a, b):
            if a == NEG_INF or b == NEG_INF:
                return NEG_INF
            # Python integers handle arbitrary size, so overflow is not an issue here.
            return a + b

        # Merge function combines the results from two child nodes (left and right)
        # to compute the state for the parent node.
        def merge(left_data, right_data):
            # Unpack the states from left and right children
            S00_l, S01_l, S10_l, S11_l = left_data
            S00_r, S01_r, S10_r, S11_r = right_data
            
            # Calculate S00 for parent: Left boundary excluded, Right boundary excluded
            # Possible combinations across the midpoint boundary M, M+1:
            # Left state ends without M, Right state starts without M+1. (S00_l, S00_r)
            # Left state ends without M, Right state starts with M+1. (S00_l, S10_r)
            # Left state ends with M, Right state starts without M+1. (S01_l, S00_r)
            S00_p = max(safe_add(S00_l, S00_r), 
                        safe_add(S00_l, S10_r), 
                        safe_add(S01_l, S00_r))
            
            # Calculate S01 for parent: Left boundary excluded, Right boundary included
            # Possible combinations: (S00_l, S01_r), (S00_l, S11_r), (S01_l, S01_r)
            S01_p = max(safe_add(S00_l, S01_r), 
                        safe_add(S00_l, S11_r), 
                        safe_add(S01_l, S01_r))

            # Calculate S10 for parent: Left boundary included, Right boundary excluded
            # Possible combinations: (S10_l, S00_r), (S10_l, S10_r), (S11_l, S00_r)
            S10_p = max(safe_add(S10_l, S00_r), 
                        safe_add(S10_l, S10_r), 
                        safe_add(S11_l, S00_r))

            # Calculate S11 for parent: Left boundary included, Right boundary included
            # Possible combinations: (S10_l, S01_r), (S10_l, S11_r), (S11_l, S01_r)
            S11_p = max(safe_add(S10_l, S01_r), 
                        safe_add(S10_l, S11_r), 
                        safe_add(S11_l, S01_r))

            # The `max` function naturally handles NEG_INF correctly.
            # E.g., max(NEG_INF, 5) = 5, max(NEG_INF, NEG_INF) = NEG_INF.
            return (S00_p, S01_p, S10_p, S11_p)

        # Recursive function to build the segment tree initially.
        def build(v, tl, tr):
            # v: current node index (1-based)
            # tl, tr: segment tree node range [tl, tr] (0-based indices of nums)
            if tl == tr:
                # Leaf node representing a single element nums[tl]
                val = nums[tl]
                # S00: Exclude this element. Max sum is 0 (empty subsequence).
                # S11: Include this element. Max sum is max(0, val) because we only add non-negative contributions.
                # S01, S10: States representing one boundary included and other excluded. Not possible for single element range L=R. Use NEG_INF.
                tree[v] = (0, NEG_INF, NEG_INF, max(0, val))
            else:
                # Internal node. Recursively build children and merge results.
                tm = (tl + tr) // 2 # Midpoint
                build(2*v, tl, tm)       # Build left child
                build(2*v + 1, tm + 1, tr) # Build right child
                tree[v] = merge(tree[2*v], tree[2*v + 1]) # Merge results

        # Recursive function to update a value in the segment tree.
        def update(v, tl, tr, pos, new_val):
            # v: current node index
            # tl, tr: segment tree node range [tl, tr]
            # pos: index in the original array nums to update (0-based)
            # new_val: the new value for nums[pos]
            if tl == tr:
                # Leaf node corresponding to pos. Update its state.
                tree[v] = (0, NEG_INF, NEG_INF, max(0, new_val))
            else:
                # Internal node. Determine which child subtree contains pos and recurse.
                tm = (tl + tr) // 2
                if pos <= tm:
                    update(2*v, tl, tm, pos, new_val)
                else:
                    update(2*v + 1, tm + 1, tr, pos, new_val)
                # After the child is updated, recompute this node's state by merging children.
                tree[v] = merge(tree[2*v], tree[2*v + 1])

        # Build the segment tree based on the initial `nums` array.
        build(1, 0, N - 1)

        total_max_sum = 0
        
        # Process each query.
        for pos, x in queries:
            # Update the segment tree with the new value at the specified position.
            update(1, 0, N - 1, pos, x)
            
            # The result for the current state of `nums` is stored at the root of the tree (node 1).
            root_data = tree[1]
            # The maximum sum subsequence is the maximum value among the four states at the root.
            current_max_sum = max(root_data)
            
            # Since the maximum sum must be non-negative (empty subsequence gives sum 0),
            # ensure the result is at least 0. This guards against potential issues if max returns NEG_INF,
            # though theoretically, it should not happen for N >= 1.
            current_max_sum = max(0, current_max_sum) 

            # Add the current query's result to the total sum, applying modulo arithmetic.
            total_max_sum = (total_max_sum + current_max_sum) % MOD
            
        # Return the final accumulated sum modulo MOD.
        return total_max_sum