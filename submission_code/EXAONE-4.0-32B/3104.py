import bisect

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        exists = [False] * (n + 1)
        for x in nums:
            if x <= n:
                exists[x] = True
        
        ways = 0
        for k in range(0, n + 1):
            if exists[k]:
                continue
            count_lt = bisect.bisect_left(sorted_nums, k)
            if count_lt == k:
                ways += 1
                
        return ways