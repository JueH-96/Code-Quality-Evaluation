class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        s = set()
        ans = 0
        for x in reversed(nums):
            ans += 1
            if 1 <= x <= k:
                s.add(x)
            if len(s) == k:
                return ans
        return ans