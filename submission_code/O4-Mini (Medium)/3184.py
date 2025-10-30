from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute A[i] = nums[i] - i
        A = [nums[i] - i for i in range(n)]
        # Coordinate compress A
        vals = sorted(set(A))
        idx_map = {v: i+1 for i, v in enumerate(vals)}  # 1-based
        
        # Fenwick tree (BIT) for range max query on prefix [1..i]
        m = len(vals)
        bit = [0] * (m + 1)
        
        def bit_update(i: int, v: int):
            while i <= m:
                if v > bit[i]:
                    bit[i] = v
                i += i & -i
        
        def bit_query(i: int) -> int:
            res = 0
            while i > 0:
                if bit[i] > res:
                    res = bit[i]
                i -= i & -i
            return res
        
        ans = -10**30
        # DP: dp[i] = nums[i] + max(0, max dp[j] where j<i and A[j] <= A[i])
        for i in range(n):
            ai = A[i]
            ci = idx_map[ai]
            best_prev = bit_query(ci)
            # best_prev is >= 0 since tree initialized with 0 (we treat empty as 0)
            dp = nums[i] + best_prev
            # update global answer
            if dp > ans:
                ans = dp
            # update BIT with dp (only if dp > 0 will it raise the stored value)
            bit_update(ci, dp)
        
        return ans