class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # prefix[i] = sum of nums[0..i-1]
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]

        # dp[i][j] = True if subarray nums[i..j] can be fully split into singletons
        dp = [[False] * n for _ in range(n)]
        # base cases: length 1 and length 2 are always splittable
        for i in range(n):
            dp[i][i] = True
        for i in range(n - 1):
            dp[i][i+1] = True

        # helper to get sum of nums[l..r]
        def segment_sum(l: int, r: int) -> int:
            return prefix[r+1] - prefix[l]

        # fill dp by increasing segment length
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                # try all possible split points k: [i..k] and [k+1..j]
                can_split = False
                for k in range(i, j):
                    lenL = k - i + 1
                    lenR = j - k
                    sumL = segment_sum(i, k)
                    sumR = segment_sum(k+1, j)
                    # check the split rule for the two resulting subarrays
                    okL = (lenL == 1) or (sumL >= m)
                    okR = (lenR == 1) or (sumR >= m)
                    if okL and okR and dp[i][k] and dp[k+1][j]:
                        can_split = True
                        break
                dp[i][j] = can_split

        # answer for the whole array
        return dp[0][n-1]