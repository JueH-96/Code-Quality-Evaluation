class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        ones = sum(nums)
        
        if ones == n:
            return 0
        if ones == 0:
            return -1 if n % 2 == 0 else (n + 1) // 2
        
        max_ones = ones
        current_ones = sum(nums[:3])
        
        for i in range(3, n):
            current_ones = current_ones - nums[i-3] + nums[i]
            max_ones = max(max_ones, current_ones)
        
        if max_ones == 0:
            return -1
        
        return n - max_ones