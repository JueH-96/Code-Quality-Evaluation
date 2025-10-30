class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        prefix = [0]*(n+1)
        suffix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = (prefix[i] + nums[i]) % mod
            suffix[i+1] = (suffix[i] + nums[n-i-1]) % mod
        count = 0
        for i in range(n):
            count = (count + prefix[i] * suffix[n-i-1]) % mod
        return count