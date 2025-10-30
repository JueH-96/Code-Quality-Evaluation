class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        max_length = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums) + 1):
                subarray = nums[i:j]
                if subarray == sorted(subarray):
                    max_length = max(max_length, len(subarray))
        return max_length