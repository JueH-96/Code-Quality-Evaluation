class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        # We'll use a greedy approach to partition the array into as many
        # "summed-up" segments as possible, with each segment's sum >= the sum
        # of the previous segment. This ensures the final sequence of sums is
        # non-decreasing and has maximal length.
        
        count = 0        # number of segments formed
        prev_sum = 0     # sum of the last finalized segment
        current_sum = 0  # building sum of the current segment
        
        for num in nums:
            current_sum += num
            # if the current segment sum is at least the previous segment sum,
            # we can finalize this segment and start a new one
            if current_sum >= prev_sum:
                count += 1
                prev_sum = current_sum
                current_sum = 0
        
        return count