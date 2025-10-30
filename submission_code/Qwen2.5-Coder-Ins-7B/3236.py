class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        sequential_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                sequential_sum += nums[i]
            else:
                break
        
        missing = sequential_sum
        while missing in nums:
            missing += 1
        
        return missing