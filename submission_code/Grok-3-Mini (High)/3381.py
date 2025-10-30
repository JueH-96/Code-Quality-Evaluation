class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        min_length = float('inf')
        n = len(nums)
        for start in range(n):
            or_val = 0
            for end in range(start, n):
                or_val |= nums[end]
                if or_val >= k:
                    length = end - start + 1
                    if length < min_length:
                        min_length = length
                    break  # No need to check further ends for this start
        if min_length == float('inf'):
            return -1
        else:
            return min_length