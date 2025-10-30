class Solution:
    def isGood(self, nums: List[int]) -> bool:
        from collections import Counter
        n = max(nums)
        base = [i for i in range(1, n)] + [n]*2
        count_nums = Counter(nums)
        count_base = Counter(base)
        return count_nums == count_base