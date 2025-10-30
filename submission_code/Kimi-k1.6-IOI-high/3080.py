class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        total_and = nums[0]
        for num in nums[1:]:
            total_and &= num
        
        if total_and != 0:
            return 1
        
        count = 0
        current_and = None
        for num in nums:
            if current_and is None:
                current_and = num
            else:
                current_and &= num
            if current_and == 0:
                count += 1
                current_and = None
        
        return count