class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        import itertools
        ans = 0
        for p in itertools.permutations(nums):
            cur = ""
            for x in p:
                cur += bin(x)[2:]
            ans = max(ans, int(cur, 2))
        return ans