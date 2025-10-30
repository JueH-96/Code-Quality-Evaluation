class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        max_or, prefix, n = 0, 0, len(nums)
        
        for num in nums:
            prefix |= num
        
        for i in range(n):
            current = prefix & ~nums[i] | (nums[i] << k)    
            max_or = max(max_or, current)
            
        return max_or