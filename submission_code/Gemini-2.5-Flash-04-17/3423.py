from typing import List

class Solution:
    MOD = 10**9 + 7
    # Use float('-inf') for mathematical infinity in max-plus algebra
    INF = float('-inf') 

    def add(self, a, b):
        # Custom add function to handle INF correctly in max-plus algebra
        # If either operand is INF, the sum is INF
        if a == self.INF or b == self.INF:
            return self.INF
        return a + b

    def multiply(self, mat1, mat2):
        """Max-plus matrix multiplication: C = mat1 * mat2"""
        # C[i][j] = max_k (mat1[i][k] + mat2[k][j])
        res = [[self.INF, self.INF], [self.INF, self.INF]]
        
        # Compute res[0][0] = max(mat1[0][0] + mat2[0][0], mat1[0][1] + mat2[1][0])
        res[0][0] = max(self.add(mat1[0][0], mat2[0][0]), self.add(mat1[0][1], mat2[1][0]))
        
        # Compute res[0][1] = max(mat1[0][0] + mat2[0][1], mat1[0][1] + mat2[1][1])
        res[0][1] = max(self.add(mat1[0][0], mat2[0][1]), self.add(mat1[0][1], mat2[1][1]))
        
        # Compute res[1][0] = max(mat1[1][0] + mat2[0][0], mat1[1][1] + mat2[1][0])
        res[1][0] = max(self.add(mat1[1][0], mat2[0][0]), self.add(mat1[1][1], mat2[1][0]))
        
        # Compute res[1][1] = max(mat1[1][0] + mat2[0][1], mat1[1][1] + mat2[1][1])
        res[1][1] = max(self.add(mat1[1][0], mat2[0][1]), self.add(mat1[1][1], mat2[1][1]))
        
        return res

    def build(self, v, tl, tr, nums):
        """Build segment tree recursively"""
        if tl == tr:
            # Leaf node for index tl with value nums[tl]
            val = nums[tl]
            # The matrix for a single element [i, i] transforms dp[i-1] to dp[i].
            # dp[i][0] = max(dp[i-1][0], dp[i-1][1])
            # dp[i][1] = dp[i-1][0] + nums[i]
            # In max-plus algebra:
            # dp[i][0] = max(dp[i-1][0] + 0, dp[i-1][1] + 0)
            # dp[i][1] = max(dp[i-1][0] + nums[i], dp[i-1][1] + INF) # Cannot include nums[i] if i-1 was included
            # This corresponds to the matrix [[0, 0], [nums[i], INF]]
            self.tree[v] = [[0, 0], [val, self.INF]]
        else:
            tm = (tl + tr) // 2
            # Recursively build left and right children
            self.build(2 * v + 1, tl, tm, nums) # Left child covering [tl, tm]
            self.build(2 * v + 2, tm + 1, tr, nums) # Right child covering [tm+1, tr]
            
            # The matrix for the combined range [tl, tr] transforms dp[tl-1] to dp[tr].
            # It's the product of the matrix for [tm+1, tr] (transforms dp[tm] to dp[tr])
            # and the matrix for [tl, tm] (transforms dp[tl-1] to dp[tm]).
            # T(tl, tr) = T(tm+1, tr) * T(tl, tm) using max-plus matrix multiplication.
            self.tree[v] = self.multiply(self.tree[2 * v + 2], self.tree[2 * v + 1])

    def update(self, v, tl, tr, pos, new_val):
        """Update the value at position pos and propagate changes up the tree"""
        if tl == tr:
            # Found the leaf node corresponding to 'pos'
            # Update the matrix for this leaf node
            self.tree[v] = [[0, 0], [new_val, self.INF]]
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                # Update in the left child's subtree
                self.update(2 * v + 1, tl, tm, pos, new_val)
            else:
                # Update in the right child's subtree
                self.update(2 * v + 2, tm + 1, tr, pos, new_val)
                
            # After a child is updated, the matrix for the parent node needs recomputation
            self.tree[v] = self.multiply(self.tree[2 * v + 2], self.tree[2 * v + 1])

    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        # The segment tree will store 2x2 matrices.
        # Tree size 4*n is sufficient for 0-based indexing (node v has children 2v+1, 2v+2)
        self.tree = [None] * (4 * n)
        
        # Build the initial segment tree based on the initial nums array
        self.build(0, 0, n - 1, nums)

        total_sum = 0
        for pos, x in queries:
            # For each query, update the element at pos and its effect in the tree
            # We don't need to modify the original nums array itself, just the tree.
            self.update(0, 0, n - 1, pos, x)

            # After the update, the root node (index 0) covers the entire range [0, n-1].
            # The matrix at the root, self.tree[0], is T(0, n-1).
            # This matrix transforms the DP state before index 0 (dp[-1])
            # to the DP state at index n-1 (dp[n-1]).
            # We use the convention dp[-1] = [0, INF]^T (cannot include an element before index 0).
            # [dp[n-1][0], dp[n-1][1]]^T = T(0, n-1) * [0, INF]^T
            # If T(0, n-1) = [[a, b], [c, d]], then
            # dp[n-1][0] = max(a + 0, b + INF) = a
            # dp[n-1][1] = max(c + 0, d + INF) = c
            # So, dp[n-1][0] is self.tree[0][0][0] and dp[n-1][1] is self.tree[0][1][0].

            dp_n_minus_1_0 = self.tree[0][0][0]
            dp_n_minus_1_1 = self.tree[0][1][0]

            # The maximum sum of a non-adjacent subsequence for the entire array is max(dp[n-1][0], dp[n-1][1]).
            # The empty subsequence is always a valid option with sum 0.
            # So the actual maximum sum is max(0, max(dp[n-1][0], dp[n-1][1])).
            current_query_sum = max(dp_n_minus_1_0, dp_n_minus_1_1)
            current_query_sum = max(0, current_query_sum) # Ensure the sum is non-negative

            # Add the current query's answer to the total sum, applying modulo 10^9 + 7.
            # Since current_query_sum is non-negative, standard modulo works.
            total_sum = (total_sum + current_query_sum) % self.MOD

        return total_sum