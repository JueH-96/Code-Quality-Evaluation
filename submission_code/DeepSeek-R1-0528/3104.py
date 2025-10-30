import bisect

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ways = 0
        for k in range(0, n + 1):
            c = bisect.bisect_left(nums, k)
            if c != k:
                continue
            if k < n:
                if nums[k] > k:
                    ways += 1
            else:
                ways += 1
        return ways