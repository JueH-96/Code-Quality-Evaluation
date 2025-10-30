class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        
        # Calculate prefix sums for O(1) subarray sum queries
        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i + 1] = prefix_sums[i] + nums[i]
        
        min_sum_above_zero = None
        
        # Check all subarrays of length between l and r
        for start in range(n):
            for end in range(start, n):
                length = end - start + 1
                if l <= length <= r:
                    current_sum = prefix_sums[end + 1] - prefix_sums[start]
                    if current_sum > 0:
                        if min_sum_above_zero is None or current_sum < min_sum_above_zero:
                            min_sum_above_zero = current_sum
        
        return min_sum_above_zero if min_sum_above_zero is not None else -1