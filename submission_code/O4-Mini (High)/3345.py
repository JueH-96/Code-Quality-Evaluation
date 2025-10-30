from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # dp[s][t] = number of ways to pick a subset of size t with sum s
        # we only need sums up to k
        dp = [[0] * (n+1) for _ in range(k+1)]
        dp[0][0] = 1
        
        # build dp
        for num in nums:
            if num > k:
                # still need to shift sizes, but sums > k are irrelevant
                # we only update dp for s >= num, so skip if num > k
                for s in range(k, -1, -1):
                    # only size dimension shifts, but sum update won't occur
                    # so we don't need to do anything here
                    pass
            else:
                # update dp in reverse to avoid reuse in same iteration
                for s in range(k, num-1, -1):
                    row_prev = dp[s-num]
                    row_cur = dp[s]
                    # for each size t, add subsets that include this num
                    # from size t-1 at previous sum
                    for t in range(n, 0, -1):
                        row_cur[t] = (row_cur[t] + row_prev[t-1]) % MOD
        
        # precompute powers of 2 up to n
        pow2 = [1] * (n+1)
        for i in range(1, n+1):
            pow2[i] = (pow2[i-1] * 2) % MOD
        
        # answer = sum over all subsets B with sum k:
        #           2^(n - |B|)
        ans = 0
        # for every possible size t of B
        for t in range(n+1):
            cnt = dp[k][t]
            if cnt:
                ans = (ans + cnt * pow2[n-t]) % MOD
        
        return ans