from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        # We transform the constraint nums[j] - nums[i] >= j - i
        # into: (nums[j] - j) >= (nums[i] - i).
        # Define A[i] = nums[i] - i. We need a subsequence where A is non-decreasing,
        # maximizing the sum of nums. This is a "max-sum non-decreasing subsequence"
        # problem on the sequence A, with weights nums.
        #
        # We do this by DP and a Fenwick Tree (BIT) supporting prefix maximum queries.
        # Let dp[i] be the best sum ending at i. Then
        # dp[i] = nums[i] + max{ dp[k] : k < i and A[k] <= A[i] }, or nums[i] if none.
        # We compress the values of A to indices 1..M and maintain a BIT for max dp.
        
        n = len(nums)
        # Build adjusted array A
        A = [nums[i] - i for i in range(n)]
        
        # Coordinate-compress A
        sorted_vals = sorted(set(A))
        comp = {v: i+1 for i, v in enumerate(sorted_vals)}  # 1-based
        
        # Fenwick for prefix max
        M = len(sorted_vals)
        bit = [0] * (M+1)
        
        def bit_update(i: int, val: int):
            # set bit[i] = max(bit[i], val), propagate
            while i <= M:
                if bit[i] < val:
                    bit[i] = val
                i += i & -i
        
        def bit_query(i: int) -> int:
            # max on [1..i]
            res = 0
            while i > 0:
                if bit[i] > res:
                    res = bit[i]
                i -= i & -i
            return res
        
        ans = -10**30
        for i in range(n):
            idx = comp[A[i]]
            best_prev = bit_query(idx)  # max dp for A[k] <= A[i]
            # If best_prev is 0, we effectively start a new subsequence here
            dp_i = best_prev + nums[i]
            # Update Fenwick
            bit_update(idx, dp_i)
            # Track global max
            if dp_i > ans:
                ans = dp_i
        
        return ans