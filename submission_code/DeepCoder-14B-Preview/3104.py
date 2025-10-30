from bisect import bisect_left

class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums_sorted = sorted(nums)
        s = set(nums)
        n = len(nums)
        total = 0
        for k in range(n + 1):
            if k in s:
                continue
            count_A = bisect_left(nums_sorted, k)
            if count_A == k:
                total += 1
        return total