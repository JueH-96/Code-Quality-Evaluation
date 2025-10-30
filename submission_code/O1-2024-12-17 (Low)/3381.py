class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        import math
        
        n = len(nums)
        min_len = math.inf
        
        for start in range(n):
            current_or = 0
            for end in range(start, n):
                current_or |= nums[end]
                if current_or >= k:
                    min_len = min(min_len, end - start + 1)
                    break
        
        return min_len if min_len != math.inf else -1