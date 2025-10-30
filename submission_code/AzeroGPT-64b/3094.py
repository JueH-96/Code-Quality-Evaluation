class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        c = Counter(nums)
        ans = 0
        for k in c.keys():
            if c[k] < 2:
                return -1
            ans += (c[k] + 2) // 3
        return ans