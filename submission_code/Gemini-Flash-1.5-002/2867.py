class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        count = 0
        ans = 1
        for num in nums:
            if num == 1:
                if count > 0:
                    ans = (ans * count) % mod
                count += 1
        return ans