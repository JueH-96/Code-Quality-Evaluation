from bisect import bisect_right, insort

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        ans = 0
        asc = [0] # ascending array for lower bound
        desc = [0] # descending array for upper bound
        
        for i in range(n):
            left = bisect_right(asc, nums[i] - 50) - 1
            right = bisect_right(desc, 50 - nums[i])
            ans = (ans + left * (i - right) + right * (len(asc) - 1 - left)) % mod
            insort(asc, nums[i])
            insort(desc, -nums[i])
        return ans