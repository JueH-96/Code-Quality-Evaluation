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
                    break  # Optimization: Once we find a special subarray starting at i, we can break the inner loop as we want the shortest subarray.
                             # We can break because extending the subarray will not give a shorter length.

        if min_length == float('inf'):
            return -1
        else:
            return min_length