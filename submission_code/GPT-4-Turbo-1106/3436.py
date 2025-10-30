class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        for i in range(len(nums)):
            or_value = 0
            for j in range(i, len(nums)):
                or_value |= nums[j]
                min_diff = min(min_diff, abs(k - or_value))
                if min_diff == 0:
                    return 0  # Early stopping if we find an exact match
        return min_diff