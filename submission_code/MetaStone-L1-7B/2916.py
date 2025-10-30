class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n == 0:
            return False
        
        # Precompute the sum of subarrays for quick access
        sum_ij = [[0] * n for _ in range(n)]
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += nums[j]
                sum_ij[i][j] = s
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def can_split(i, j):
            if i == j:
                return True
            for k in range(i, j):
                left_sum = sum_ij[i][k]
                right_sum = sum_ij[k+1][j]
                
                # Check if left part is valid
                left_ok = (left_sum >= m) or (k - i + 1 == 1)
                # Check if right part is valid
                right_ok = (right_sum >= m) or ((j - (k + 1) + 1) == 1)
                
                if left_ok and right_ok:
                    if can_split(i, k) and can_split(k+1, j):
                        return True
            return False
        
        return can_split(0, n-1)