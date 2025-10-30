class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Calculate the prefix sums
        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i + 1] = prefix_sums[i] + nums[i]
        
        # To store the maximum strength
        max_strength = float('-inf')
        
        # Iterate over all possible subarrays
        for i in range(n):
            for j in range(i, n):
                subarray_sum = prefix_sums[j + 1] - prefix_sums[i]
                # Calculate the strength for the current subarray
                strength = subarray_sum * (k - 1) + subarray_sum * k
                max_strength = max(max_strength, strength)
        
        return max_strength