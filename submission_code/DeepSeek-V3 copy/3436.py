class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        n = len(nums)
        for i in range(n):
            current_or = nums[i]
            for j in range(i, n):
                current_or |= nums[j]
                current_diff = abs(current_or - k)
                if current_diff < min_diff:
                    min_diff = current_diff
                if min_diff == 0:
                    return 0
        return min_diff