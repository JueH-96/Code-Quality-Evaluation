class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        
        # The score of a subarray is the bitwise AND of its elements. We want to
        # split the array to minimize the sum of scores, and among such splits,
        # find the one with the maximum number of subarrays.

        # The minimum possible sum of scores is the bitwise AND of the entire array.
        # If this total AND is > 0, the minimum sum is achieved only by taking the
        # whole array as one subarray. The answer is 1.
        # If the total AND is 0, the minimum sum is 0. This is achieved by
        # partitioning the array into subarrays, each with a score of 0. To maximize
        # the number of subarrays, we greedily find the shortest possible subarrays
        # with a score of 0.

        # We can implement this with a single greedy pass.
        
        # In Python's two's complement, -1 has all bits set to 1.
        # `x & -1` equals `x`, so -1 is the identity for bitwise AND.
        current_and_value = -1
        
        subarray_count = 0
        
        for num in nums:
            current_and_value &= num
            
            # If the bitwise AND of the current segment becomes 0, we have found a
            # valid subarray for our partition that contributes 0 to the sum.
            if current_and_value == 0:
                subarray_count += 1
                # Reset to start a new subarray segment.
                current_and_value = -1
        
        # If `subarray_count` is 0, it means the AND of the whole array is non-zero.
        # This falls into the first case, and the answer must be 1.
        # If `subarray_count > 0`, the greedy strategy has found the maximum
        # possible number of zero-score subarrays.
        # The expression `max(1, subarray_count)` correctly handles both cases.
        return max(1, subarray_count)