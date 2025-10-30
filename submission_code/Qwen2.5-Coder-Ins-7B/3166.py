class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        from collections import Counter
        count = Counter(nums)
        n = len(nums)
        for k in range(n // 2, 0, -1):
            groups = 0
            for v in count.values():
                groups += (v + k - 1) // k
            if groups <= n // 2 + 1:
                return groups