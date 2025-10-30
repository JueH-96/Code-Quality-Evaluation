from bisect import bisect_right
from typing import List

MOD = 10**9 + 7

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        # We first sort the array.
        a = sorted(nums)
        n = len(a)
        
        # All possible differences between any two array elements (adjacent in a chosen subsequence
        # will have one of these differences as their gap, and the minimum gap equals one of these values).
        candidates = set()
        for i in range(n):
            for j in range(i+1, n):
                candidates.add(a[j] - a[i])
        cand_list = sorted(candidates)
        
        # For a given threshold t, define F(t) = number of subsequences (of k elements taken in sorted order)
        # such that every adjacent gap (a[i+1]-a[i]) is at least t.
        # Then if a subsequence’s power = min_gap, one easily sees that it will be counted in F(t)
        # for every t <= min_gap. In particular, the number of subsequences with minimum gap exactly d
        # is F(d) - F(d+1). (For t larger than any possible gap, F(t) will be 0.)
        #
        # We now write a DP routine (with caching) for F(threshold).
        
        count_cache = {}
        def F(threshold: int) -> int:
            if threshold in count_cache:
                return count_cache[threshold]
            # dp[r][i] = number of subsequences of length (r+1) that end at index i.
            # Here, r==0 corresponds to subsequences of length 1.
            dp = [[0] * n for _ in range(k)]
            # Base case: subsequences of length 1: each element is counted.
            for i in range(n):
                dp[0][i] = 1
            # For subsequences of length r+1 (r from 1 to k-1)
            for r in range(1, k):
                # Build prefix sums for dp[r-1] so we can aggregate sums quickly.
                prefix = [0] * n
                prefix[0] = dp[r-1][0]
                for i in range(1, n):
                    prefix[i] = prefix[i-1] + dp[r-1][i]
                # For each position i, we add contributions from earlier indices j
                # that satisfy a[i] - a[j] >= threshold.
                for i in range(n):
                    # Because ‘a’ is sorted, a[j] <= a[i] - threshold iff j is at most pos.
                    pos = bisect_right(a, a[i] - threshold) - 1
                    dp[r][i] = prefix[pos] if pos >= 0 else 0
            total = sum(dp[k-1])
            count_cache[threshold] = total
            return total
        
        # Now, the sum we need is the sum over all valid subsequences of
        # (minimum adjacent gap). Since the minimum gap of any subsequence is always
        # one of the differences between some two numbers in the array,
        # we can “discretize” and write:
        #
        #      answer = sum_{d in candidates} d * (F(d) - F(d+1))
        #
        # (Here F(x) means the number of subsequences with all adjacent gaps >= x;
        # note that for x larger than the maximum gap, F(x)=0.)
        
        ans = 0
        for d in cand_list:
            count_d = F(d)
            count_d_next = F(d + 1)
            # The subsequences with minimum gap exactly d.
            exact = count_d - count_d_next
            ans = (ans + d * exact) % MOD
        return ans % MOD


# Example testing code
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.sumOfPowers([1,2,3,4], 3))  # Expected output: 4
    # Example 2:
    print(sol.sumOfPowers([2,2], 2))       # Expected output: 0
    # Example 3:
    print(sol.sumOfPowers([4,3,-1], 2))    # Expected output: 10