from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        # Transform value A[i] = nums[i] - i
        A = [nums[i] - i for i in range(n)]
        # Coordinate compress A
        sorted_A = sorted(set(A))
        comp = {v: i + 1 for i, v in enumerate(sorted_A)}  # 1-based ranks
        m = len(sorted_A)
        
        # Fenwick tree (BIT) for range prefix-max queries
        NEG_INF = -10**30
        ft = [NEG_INF] * (m + 1)
        
        def ft_update(idx: int, val: int) -> None:
            # point-update: ft[idx] = max(ft[idx], val)
            while idx <= m:
                if ft[idx] < val:
                    ft[idx] = val
                idx += idx & -idx
        
        def ft_query(idx: int) -> int:
            # prefix query: max over ft[1..idx]
            res = NEG_INF
            while idx > 0:
                if ft[idx] > res:
                    res = ft[idx]
                idx -= idx & -idx
            return res
        
        ans = NEG_INF
        for i in range(n):
            rank = comp[A[i]]
            prev_max = ft_query(rank)
            # Either extend a previous subsequence or start new
            if prev_max > 0:
                dp = nums[i] + prev_max
            else:
                dp = nums[i]
            # Update Fenwick tree and global answer
            ft_update(rank, dp)
            if dp > ans:
                ans = dp
        
        return ans