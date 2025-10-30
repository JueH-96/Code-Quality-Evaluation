class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        n = len(nums)
        longest_prefix_sum = nums[0]
        longest_prefix_len = 1
        for i in range(1, n):
            if nums[i] == nums[i-1] + 1:
                longest_prefix_sum += nums[i]
                longest_prefix_len += 1
            else:
                break
        
        missing = longest_prefix_sum
        nums_set = set(nums)
        while missing in nums_set:
            missing += 1
        return missing