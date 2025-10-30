class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        nums.sort()
        max_freq = 0
        count = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                count += 1
            else:
                max_freq = max(max_freq, count)
                count = 1
        max_freq = max(max_freq, count)
        return max_freq