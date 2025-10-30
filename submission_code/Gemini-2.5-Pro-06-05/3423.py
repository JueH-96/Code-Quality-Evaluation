import math
from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        
        MOD = 10**9 + 7
        N = len(nums)
        NEG_INF = -float('inf')

        # The segment tree stores 2x2 matrices for (max, +) algebra.
        # A node's matrix represents the transformation for its range.
        # State vector: [dp_not_taken, dp_taken]
        # Transition from i-1 to i:
        # dp[i][0] = max(dp[i-1][0], dp[i-1][1])
        # dp[i][1] = dp[i-1][0] + nums[i]
        # This corresponds to multiplying by the matrix T_i = [[0, 0], [nums[i], -inf]]
        tree = [[[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]] for _ in range(4 * N)]

        def mat_mul(A, B):
            """Performs matrix multiplication: C = A * B in (max, +) algebra."""
            C = [[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]
            C[0][0] = max(A[0][0] + B[0][0], A[0][1] + B[1][0])
            C[0][1] = max(A[0][0] + B[0][1], A[0][1] + B[1][1])
            C[1][0] = max(A[1][0] + B[0][0], A[1][1] + B[1][0])
            C[1][1] = max(A[1][0] + B[0][1], A[1][1] + B[1][1])
            return C

        def build(node, start, end):
            if start == end:
                # Leaf node matrix for nums[start]
                val = nums[start]
                tree[node] = [[0, 0], [val, NEG_INF]]
                return

            mid = (start + end) // 2
            build(2 * node, start, mid)
            build(2 * node + 1, mid + 1, end)
            
            # Combine children: T_parent = T_right * T_left
            tree[node] = mat_mul(tree[2 * node + 1], tree[2 * node])

        def update(node, start, end, idx, val):
            if start == end:
                # Update leaf node matrix
                tree[node] = [[0, 0], [val, NEG_INF]]
                return

            mid = (start + end) // 2
            if idx <= mid:
                update(2 * node, start, mid, idx, val)
            else:
                update(2 * node + 1, mid + 1, end, idx, val)
            
            # Re-combine children
            tree[node] = mat_mul(tree[2 * node + 1], tree[2 * node])

        # Initial build of the segment tree
        build(1, 0, N - 1)

        total_sum = 0
        for pos, x in queries:
            update(1, 0, N - 1, pos, x)
            
            root_matrix = tree[1]
            
            # The final DP state V_final is root_matrix * V_initial, where V_initial = [0, -inf]^T.
            # This results in V_final = [root_matrix[0][0], root_matrix[1][0]]^T.
            dp_not_taken = root_matrix[0][0]
            dp_taken = root_matrix[1][0]
            
            # The answer is the maximum possible sum, including the option of an empty subsequence (sum 0).
            query_ans = max(0, dp_not_taken, dp_taken)
            
            total_sum += query_ans
        
        return total_sum % MOD