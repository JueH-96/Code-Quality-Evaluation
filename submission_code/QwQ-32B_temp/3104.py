import bisect

class Solution:
    def countWays(self, nums):
        nums.sort()
        n = len(nums)
        ans = 0
        for k in range(n + 1):
            pos = bisect.bisect_left(nums, k)
            if pos != k:
                continue
            # Check if there's any element equal to k
            if pos < n and nums[pos] == k:
                continue
            ans += 1
        return ans