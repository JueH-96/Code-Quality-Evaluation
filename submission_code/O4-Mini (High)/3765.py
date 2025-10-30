from typing import List

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        # prefix sums of nums
        Psum = [0] * n
        Psum[0] = nums[0]
        for i in range(1, n):
            Psum[i] = Psum[i-1] + nums[i]
        # prefix sums of cost
        prefA = [0] * n
        prefA[0] = cost[0]
        for i in range(1, n):
            prefA[i] = prefA[i-1] + cost[i]
        # prefix sums of cost[j] * Psum[j]
        prefB = [0] * n
        prefB[0] = cost[0] * Psum[0]
        for i in range(1, n):
            prefB[i] = prefB[i-1] + cost[i] * Psum[i]
        # Compute the constant parts:
        # constant1 = sum_{j=0..n-1} cost[j] * Psum[j]
        constant1 = prefB[n-1]
        # constant2 = k * sum_{j=0..n-1} cost[j]
        constant2 = k * prefA[n-1]
        # penalty for cutting at position q (before q), for q in [1..n-1]. At q=n or q=0, penalty=0.
        # suffixCostSum[q] = sum cost[j] for j>=q = prefA[n-1] - prefA[q-1]
        penalty = [0] * (n + 1)
        totalCostSum = prefA[n-1]
        for q in range(1, n):
            penalty[q] = k * (totalCostSum - prefA[q-1])
        penalty[n] = 0  # no penalty if segment ends at n

        # dp[r] = minimal extra cost X(S) for segmenting from r..n-1, assuming r is segment-start
        dp = [0] * (n + 1)
        dp[n] = 0
        INF = 10**30
        # Iterate r from n-1 down to 0
        for r in range(n-1, -1, -1):
            best = INF
            # try next cut at q, segment is [r..q-1]
            # q runs from r+1 to n inclusive
            for q in range(r+1, n+1):
                # compute W1(r,q):
                # W1 = sum_{j=r..q-2} cost[j] * (Psum[q-1] - Psum[j])
                #      = SCost * Psum[q-1] - SCoef
                # where SCost = sum cost[j], SCoef = sum cost[j]*Psum[j], j from r..q-2
                if q >= r+2:
                    # there is at least one j in [r..q-2]
                    end = q-2
                    sumA = prefA[end] - (prefA[r-1] if r > 0 else 0)
                    sumB = prefB[end] - (prefB[r-1] if r > 0 else 0)
                else:
                    sumA = 0
                    sumB = 0
                w1 = sumA * Psum[q-1] - sumB
                # add penalty for cut at q (if q<n) and dp[q]
                cost_candidate = w1 + penalty[q] + dp[q]
                if cost_candidate < best:
                    best = cost_candidate
            dp[r] = best

        # total minimum cost = constant1 + constant2 + dp[0]
        return constant1 + constant2 + dp[0]