class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        for i in range(len(nums)):
            current_or = 0
            for j in range(i, len(nums)):
                current_or |= nums[j]
                min_diff = min(min_diff, abs(k - current_or))
        return min_diff