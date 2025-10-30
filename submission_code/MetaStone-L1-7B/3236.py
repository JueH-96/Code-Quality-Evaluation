class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        max_len = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                max_len += 1
            else:
                break
        sum_val = sum(nums[:max_len])
        x = sum_val
        while x in nums:
            x += 1
        return x