class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        import math

        n = len(nums)
        m = len(andValues)

        # If we need more subarrays than elements, impossible
        if m > n:
            return -1

        # dp[i][j] will hold the minimum sum of the values of j subarrays
        # covering the first i elements of nums (i.e., up to index i - 1),
        # if it's possible. Otherwise it will be math.inf.
        dp = [[math.inf]*(m+1) for _ in range(n+1)]
        dp[0][0] = 0  # 0 elements with 0 subarrays => sum = 0

        for i in range(1, n+1):
            for j in range(1, min(i, m)+1):
                targetAND = andValues[j-1]  # we want the j-th subarray to have AND == targetAND

                rollingAND = (1 << 20) - 1  # since nums[i] < 10^5, up to ~17 bits. Start with all 1-bits set.
                # We'll try to form the subarray for the j-th piece ending at index i-1
                # Move backwards to see if we can get exactly "targetAND"
                for start in range(i-1, j-2, -1):
                    rollingAND &= nums[start]
                    # If we've dropped below the target in bits we need (rollingAND < targetAND),
                    # there's no way to get bits back. Break to prune.
                    if rollingAND < targetAND:
                        break
                    if rollingAND == targetAND:
                        # If the subarray [start..i-1] has AND == targetAND
                        # then dp[i][j] can come from dp[start][j-1], plus nums[i-1] as the value of last element
                        dp[i][j] = min(dp[i][j], dp[start][j-1] + nums[i-1])

        ans = dp[n][m]
        return ans if ans != math.inf else -1