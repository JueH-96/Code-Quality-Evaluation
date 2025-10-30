class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = (prefix[i] + nums[i] % modulo) % modulo

        from collections import defaultdict
        count = defaultdict(int)
        count[0] = 1
        ans = 0
        for i in range(1, n + 1):
            ans += count[prefix[i]]
            count[prefix[i]] += 1
            if i - modulo >= 0:
                count[prefix[i - modulo]] -= 1
        return ans