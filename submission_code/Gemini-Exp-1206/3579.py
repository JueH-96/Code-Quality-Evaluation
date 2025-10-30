class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        import itertools
        ans = 0
        for p in itertools.permutations(nums):
            s = ""
            for x in p:
                s += bin(x)[2:]
            ans = max(ans, int(s, 2))
        return ans