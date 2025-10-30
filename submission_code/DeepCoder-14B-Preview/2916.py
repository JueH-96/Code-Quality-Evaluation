class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n == 0:
            return False
        
        # Precompute the prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def helper(start, end):
            l = end - start + 1
            if l == 1:
                return True
            
            for k in range(start, end):
                left_len = k - start + 1
                right_len = end - (k + 1) + 1
                sum_left = prefix[k + 1] - prefix[start]
                sum_right = prefix[end + 1] - prefix[k + 1]
                
                # Check if both parts satisfy the condition
                if (left_len == 1 or sum_left >= m) and (right_len == 1 or sum_right >= m):
                    if helper(start, k) and helper(k + 1, end):
                        return True
            return False
        
        return helper(0, n - 1)