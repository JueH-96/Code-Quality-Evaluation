class Solution:
    def numberOfGoodSubarraySplits(self, nums):
        mod = 10**9 + 7
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        count = [0] * len(prefix)
        count[0] = 1
        for i in range(1, len(prefix)):
            count[prefix[i]] += 1
        return (sum(v * (v - 1) // 2 for v in count) % mod)