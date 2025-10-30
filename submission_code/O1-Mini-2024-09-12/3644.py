class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        prefix_sum = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        min_sum = float('inf')
        n = len(nums)
        for i in range(n):
            for size in range(l, r + 1):
                if i + size <= n:
                    current_sum = prefix_sum[i + size] - prefix_sum[i]
                    if current_sum > 0 and current_sum < min_sum:
                        min_sum = current_sum

        return min_sum if min_sum != float('inf') else -1