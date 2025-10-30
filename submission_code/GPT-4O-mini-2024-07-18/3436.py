class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_diff = float('inf')

        for l in range(n):
            current_or = 0
            for r in range(l, n):
                current_or |= nums[r]
                min_diff = min(min_diff, abs(k - current_or))
                if min_diff == 0:  # Early exit if we find an exact match
                    return 0

        return min_diff