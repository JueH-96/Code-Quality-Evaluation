class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
            suffix[n - i] = suffix[n - i + 1] + nums[n - i - 1]
        max_sum = float('-inf')
        for x in range(1, n + 1):
            left = prefix[x]
            right = suffix[x]
            curr_sum = left * x - right * (x - 1)
            max_sum = max(max_sum, curr_sum)
        return max_sum