class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        prev = -1
        ans = 1
        for i in range(n):
            if nums[i] == 1:
                if prev != -1:
                    ans = (ans * (i - prev)) % MOD
                prev = i
        return 0 if prev == -1 else ans