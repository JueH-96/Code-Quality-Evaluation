class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        min_diff = float('inf')
        n = len(nums)
        for i in range(n):
            for j in range(n):
                if abs(i - j) >= x:
                    diff = abs(nums[i] - nums[j])
                    min_diff = min(min_diff, diff)
        return min_diff