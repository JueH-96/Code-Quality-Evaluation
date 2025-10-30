from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        # Count how many times each value appears
        cnt = [0] * (n + 1)
        for x in nums:
            # x is guaranteed 0 <= x < n
            cnt[x] += 1

        ways = 0
        less = 0  # less will hold count of nums < k as we increase k

        # Check for each k = 0..n
        # Condition for k to be valid:
        #   1) exactly k elements are < k  => less == k
        #   2) no element is == k         => cnt[k] == 0
        for k in range(0, n + 1):
            if less == k and cnt[k] == 0:
                ways += 1
            # update less for next k: include those == k
            if k < n:
                less += cnt[k]

        return ways