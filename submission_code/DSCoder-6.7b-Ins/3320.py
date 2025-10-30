class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        from collections import Counter
        count = Counter(nums)
        ops = 0
        for k, v in count.items():
            ops += v // 2
        return ops