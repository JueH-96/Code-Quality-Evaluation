class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        total_and = nums[0]
        current_and = nums[0]
        count = 0
        
        for num in nums[1:]:
            current_and &= num
            if current_and == 0:
                count += 1
                current_and = num
                total_and = num
        
        if total_and == 0:
            count += 1
        
        return count