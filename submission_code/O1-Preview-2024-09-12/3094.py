class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        counts = Counter(nums)
        total_ops = 0
        for cnt in counts.values():
            if cnt == 1:
                return -1
            total_ops += (cnt + 2) // 3
        return total_ops