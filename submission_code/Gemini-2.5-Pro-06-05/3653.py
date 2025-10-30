import itertools

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        """
        Calculates the maximum sum of a subarray whose length is divisible by k.
        """
        
        n = len(nums)
        
        # We use prefix sums to efficiently calculate the sum of any subarray.
        # prefix_sums[i] stores the sum of elements from nums[0] to nums[i-1].
        # The sum of subarray nums[j..i-1] is prefix_sums[i] - prefix_sums[j].
        # We use itertools.accumulate for a concise way to generate prefix sums.
        # [0] is prepended to represent the sum of an empty prefix.
        prefix_sums = [0] + list(itertools.accumulate(nums))
        
        # The problem is to maximize prefix_sums[i] - prefix_sums[j]
        # such that the length (i-j) is a multiple of k, and j < i.
        # The condition (i - j) % k == 0 is equivalent to i % k == j % k.
        
        # For a fixed `i`, to maximize the difference, we need to minimize `prefix_sums[j]`.
        # So, we iterate through `i` and for each `i`, find the minimum `prefix_sums[j]`
        # over all `j < i` that have the same remainder `j % k == i % k`.
        
        max_sum = float('-inf')
        
        # This dictionary stores the minimum prefix sum found so far for each remainder modulo k.
        # Key: remainder, Value: minimum prefix sum.
        min_prefix_for_rem = {}
        
        for i, p_sum in enumerate(prefix_sums):
            remainder = i % k
            
            # If we've seen this remainder before, there exists an index j < i
            # such that j % k == remainder. We can form a valid subarray.
            if remainder in min_prefix_for_rem:
                # The difference between the current prefix sum and the minimum prefix sum
                # with the same remainder gives a candidate for the maximum sum.
                min_p_for_rem = min_prefix_for_rem[remainder]
                current_subarray_sum = p_sum - min_p_for_rem
                max_sum = max(max_sum, current_subarray_sum)
            
            # Update the dictionary with the minimum prefix sum for the current remainder.
            # This ensures that for subsequent indices, P[i] is a candidate for min_P[j].
            min_prefix_for_rem[remainder] = min(min_prefix_for_rem.get(remainder, float('inf')), p_sum)
            
        return max_sum