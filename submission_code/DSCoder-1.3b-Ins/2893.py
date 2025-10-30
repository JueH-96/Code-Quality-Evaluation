class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        even = [0] * (n + 1)
        odd = [0] * (n + 1)
        even[1] = nums[0]
        odd[1] = nums[0] if nums[0] % 2 == 1 else 0
        for i in range(1, n):
            even[i + 1] = max(even[i], odd[i] + nums[i])
            odd[i + 1] = max(odd[i], even[i] + nums[i]) if nums[i] % 2 == 1 else max(odd[i], even[i] + nums[i])
        return max(dp[n], even[n], odd[n])