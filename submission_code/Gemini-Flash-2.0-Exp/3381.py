class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        min_len = float('inf')
        for i in range(len(nums)):
            current_or = 0
            for j in range(i, len(nums)):
                current_or |= nums[j]
                if current_or >= k:
                    min_len = min(min_len, j - i + 1)
                    break
        if min_len == float('inf'):
            return -1
        else:
            return min_len