import bisect

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        for k in range(n + 1):
            count_less = bisect.bisect_left(nums, k)
            count_equal = bisect.bisect_right(nums, k) - count_less
            if count_less == k and count_equal == 0:
                ans += 1
        return ans