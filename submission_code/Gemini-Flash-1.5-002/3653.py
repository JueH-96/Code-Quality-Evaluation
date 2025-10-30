class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = -float('inf')

        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                sub_array_length = j - i + 1
                if sub_array_length % k == 0:
                    max_sum = max(max_sum, current_sum)

        if max_sum == -float('inf'):
            return 0 #Handle case where no subarray length is divisible by k.

        return max_sum