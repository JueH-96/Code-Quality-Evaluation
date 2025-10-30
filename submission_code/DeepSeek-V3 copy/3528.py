class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        max_score = 0
        current_max = nums[0]
        for i in range(1, n):
            max_score = max(max_score, current_max * (n - 1 - i + 1))
            current_max = max(current_max, nums[i])
        return max_score