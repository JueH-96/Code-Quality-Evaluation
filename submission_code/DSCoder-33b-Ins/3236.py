class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        nums = sorted(set(nums))
        prefix_sum = 0
        for num in nums:
            if num > prefix_sum + 1:
                break
            prefix_sum += num
        return prefix_sum + 1