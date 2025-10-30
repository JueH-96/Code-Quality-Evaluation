class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = {0: -1}
        current_sum = 0
        max_sum = float('-inf')

        for i in range(len(nums)):
            current_sum += nums[i]
            remainder = current_sum % k

            if remainder in prefix_sum:
                if (i - prefix_sum[remainder]) % k == 0:
                    max_sum = max(max_sum, current_sum - prefix_sum[remainder])
            else:
                prefix_sum[remainder] = current_sum

        return max_sum if max_sum != float('-inf') else 0