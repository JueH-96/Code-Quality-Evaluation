class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] | nums[i]
        min_length = float('inf')
        for i in range(n):
            for j in range(i, n):
                if prefix_sum[j + 1] - prefix_sum[i] >= k:
                    min_length = min(min_length, j - i + 1)
        return min_length if min_length != float('inf') else -1