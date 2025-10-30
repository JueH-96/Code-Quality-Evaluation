class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        min_length = float('inf')
        prefix_sum = 0
        sum_indices = {0: -1}

        for i in range(2 * n):
            prefix_sum += nums[i % n]
            if prefix_sum - target in sum_indices:
                min_length = min(min_length, i - sum_indices[prefix_sum - target])
            sum_indices[prefix_sum] = i

        return min_length if min_length != float('inf') else -1