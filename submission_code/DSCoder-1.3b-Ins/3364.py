class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        dp = [0] * (1 << m)
        for S in range(1 << m):
            if bin(S).count('1') % 2 == 1:
                continue
            subarray = [nums[i] for i in range(n) if (S >> i) & 1]
            if all(num & andValues[j] == andValues[j] for j, num in enumerate(subarray)):
                dp[S] = min(dp[S & (S << 1)] + sum(subarray) for S in range(1 << m))
        return min(dp) if dp[-1] != 0 else -1