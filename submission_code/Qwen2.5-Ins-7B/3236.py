class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        if not nums:
            return 1
        
        current = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                current += nums[i]
            else:
                break
        
        prefix_sum = current
        nums_set = set(nums)
        i = prefix_sum
        while i in nums_set:
            i += 1
        
        return i