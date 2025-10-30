class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        # We want to partition the array into the maximum number of contiguous segments
        # such that the sums of these segments form a non-decreasing sequence.
        #
        # Greedy approach:
        # 1. Keep track of the sum of the last finalized segment (prev_sum).
        # 2. Accumulate values in current_sum for the current segment.
        # 3. Whenever current_sum >= prev_sum, we can finalize (close) this segment,
        #    increment the count of segments, update prev_sum to current_sum,
        #    and reset current_sum to 0.
        #
        # This ensures:
        # - Each segment sum is >= the sum of the previous segment (non-decreasing).
        # - We cut as soon as it is valid, maximizing the total number of segments.
        
        segments = 0
        prev_sum = 0
        current_sum = 0
        
        for x in nums:
            current_sum += x
            # If the current segment's sum is at least the previous segment sum,
            # we can finalize this segment and start a new one.
            if current_sum >= prev_sum:
                segments += 1
                prev_sum = current_sum
                current_sum = 0
        
        return segments