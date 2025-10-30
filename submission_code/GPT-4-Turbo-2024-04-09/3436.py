class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        n = len(nums)
        
        for start in range(n):
            or_value = 0
            for end in range(start, n):
                or_value |= nums[end]
                current_diff = abs(k - or_value)
                if current_diff < min_diff:
                    min_diff = current_diff
                if min_diff == 0:
                    return 0  # Early exit if we find the exact match
        
        return min_diff