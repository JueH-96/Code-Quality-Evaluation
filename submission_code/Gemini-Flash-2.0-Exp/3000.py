class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        min_diff = float('inf')
        for i in range(len(nums)):
            for j in range(len(nums)):
                if abs(i - j) >= x:
                    min_diff = min(min_diff, abs(nums[i] - nums[j]))
        return min_diff