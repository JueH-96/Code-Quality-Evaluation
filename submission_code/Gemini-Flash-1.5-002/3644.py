class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        min_sum = float('inf')
        n = len(nums)
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                sub_array_length = j - i + 1
                if l <= sub_array_length <= r and current_sum > 0:
                    min_sum = min(min_sum, current_sum)
        if min_sum == float('inf'):
            return -1
        else:
            return min_sum