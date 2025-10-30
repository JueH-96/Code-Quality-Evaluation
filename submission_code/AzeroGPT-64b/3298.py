class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = [[1, nums[0]] for _ in range(len(nums))]
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] - dp[j][1] <= 1:
                    if dp[j][0] + 1 > dp[i][0]:
                        dp[i][0] = dp[j][0] + 1
                        dp[i][1] = nums[i]
                else:
                    break
        return max(list(zip(*dp))[0])