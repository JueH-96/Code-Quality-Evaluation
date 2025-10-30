class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        pos = [n for n in nums if n > 0]
        neg = [n for n in nums if n < 0]
        neg.sort()
        if len(neg) % 2 == 1:
            neg.pop()
        if len(pos) + len(neg) == 0:
            return max(nums)
        return -1 * reduce(lambda x, y: x * y, neg) * reduce(lambda x, y: x * y, pos, 1)