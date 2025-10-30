class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        min_length = float('inf')
        for i in range(len(nums)):
            current_or = 0
            for j in range(i, len(nums)):
                current_or |= nums[j]
                if current_or >= k:
                    min_length = min(min_length, j - i + 1)
                    break
        if min_length == float('inf'):
            return -1
        else:
            return min_length