class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        n = len(nums)
        for l in range(n):
            current_or = 0
            for r in range(l, n):
                current_or |= nums[r]
                diff = abs(k - current_or)
                min_diff = min(min_diff, diff)
        return min_diff