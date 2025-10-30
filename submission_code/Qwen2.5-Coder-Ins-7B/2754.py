class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        if len(nums) == 1:
            return nums[0]
        
        positive = [num for num in nums if num > 0]
        negative = [num for num in nums if num < 0]
        
        if not positive and len(negative) <= 1:
            return max(nums)
        
        negative.sort()
        if len(negative) % 2 != 0:
            negative.pop()
        
        return math.prod(positive + negative)