from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        groups = defaultdict(int)
        for i in range(1, n + 1):
            s = i
            f = 2
            while f * f <= s:
                while s % (f * f) == 0:
                    s //= (f * f)
                f += 1
            groups[s] += nums[i - 1]
        return max(groups.values())