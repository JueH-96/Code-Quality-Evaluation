from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        # dp[s][j] = number of subsets of size j with sum exactly s
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        
        for a in nums:
            if a > k:
                continue
            # iterate sizes and sums in reverse to avoid reuse
            for s in range(k, a - 1, -1):
                row_prev = dp[s - a]
                row_cur = dp[s]
                # for each possible size j, add subsets of size j-1
                for j in range(1, n + 1):
                    row_cur[j] = (row_cur[j] + row_prev[j - 1]) % mod
        
        # precompute powers of 2 up to n
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = (pow2[i - 1] << 1) % mod
        
        # sum over all subsets S with sum=k, weighted by 2^(n - |S|)
        ans = 0
        for j in range(n + 1):
            cnt = dp[k][j]
            if cnt:
                ans = (ans + cnt * pow2[n - j]) % mod
        
        return ans