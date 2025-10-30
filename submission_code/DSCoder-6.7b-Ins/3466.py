class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        from collections import defaultdict
        count = defaultdict(int)
        for i in range(n + 1):
            count[prefix[i]] += 1

        res = 0
        for i in range(n):
            if prefix[i] >= k:
                res += count[prefix[i] - k]

        return res