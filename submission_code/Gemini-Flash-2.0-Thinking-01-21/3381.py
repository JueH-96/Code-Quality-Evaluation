class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        min_length = float('inf')
        n = len(nums)
        for i in range(n):
            current_or = 0
            for j in range(i, n):
                current_or |= nums[j]
                if current_or >= k:
                    min_length = min(min_length, j - i + 1)
                    break # Optimization: once we find a special subarray starting at i, we can stop extending it for this starting index as we are looking for shortest length
        if min_length == float('inf'):
            return -1
        else:
            return min_length