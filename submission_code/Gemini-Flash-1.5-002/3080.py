class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        count = 1
        current_and = nums[0]
        for i in range(1, len(nums)):
            next_and = current_and & nums[i]
            if next_and == 0:
                count += 1
                current_and = nums[i]
            else:
                current_and = next_and
        return count