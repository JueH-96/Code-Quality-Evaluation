from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # build prefix sums: P[i] = sum of nums[0..i-1]
        P = [0] * (n + 1)
        for i in range(1, n + 1):
            P[i] = P[i - 1] + nums[i - 1]
        
        ans = -10**30  # sufficiently small
        # for each residue class modulo k
        for r in range(k):
            # start index in P
            i = r
            if i > n:
                break
            # initialize min prefix sum in this class
            min_pref = P[i]
            # jump in steps of k
            j = i + k
            while j <= n:
                # candidate subarray sum for length divisible by k
                candidate = P[j] - min_pref
                if candidate > ans:
                    ans = candidate
                # update the min prefix seen so far in this class
                if P[j] < min_pref:
                    min_pref = P[j]
                j += k
        
        return ans