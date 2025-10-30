class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        import math

        n, m = len(nums), len(andValues)

        # dp[i][j] = minimum possible sum of subarray values using j subarrays
        #            from the prefix nums[:i].  (i ranges from 0..n, j from 0..m)
        # We want dp[n][m] in the end; if it's still infinity, return -1
        INF = math.inf
        dp = [[INF]*(m+1) for _ in range(n+1)]
        dp[0][0] = 0  # zero elements, zero subarrays, sum = 0

        for i in range(n+1):               # i is the current "start" index in nums
            for j in range(m):             # j is how many subarrays have already been formed
                if dp[i][j] == INF:
                    continue
                # We have dp[i][j], try to form the (j+1)-th subarray
                target_and = andValues[j]  # we want the subarray AND to match this
                cur_and = (1 << 17) - 1    # since nums[i] < 10^5, use a mask that covers relevant bits
                
                # Expand subarray from i..k and update cur_and
                for k in range(i, n):
                    cur_and &= nums[k]
                    if cur_and == target_and:
                        # We can form a subarray [i..k] whose AND == target_and
                        # The value of this subarray is nums[k] (last element)
                        dp[k+1][j+1] = min(dp[k+1][j+1], dp[i][j] + nums[k])
                    # If cur_and becomes 0 but target_and != 0, no need to continue
                    # because the AND will stay 0 if we extend further
                    if cur_and == 0 and target_and != 0:
                        break
        
        ans = dp[n][m]
        return ans if ans != INF else -1