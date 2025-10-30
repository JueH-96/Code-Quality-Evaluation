class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left_or = [0] * n
        right_or = [0] * n
        
        current_or = 0
        for i in range(n):
            left_or[i] = current_or
            current_or |= nums[i]
        
        current_or = 0
        for i in range(n-1, -1, -1):
            right_or[i] = current_or
            current_or |= nums[i]
        
        max_or = 0
        for i in range(n):
            max_or = max(max_or, left_or[i] | right_or[i])
        
        return max_or