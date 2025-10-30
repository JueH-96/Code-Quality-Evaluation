class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)

        # Step 1: Precompute the last occurrence index for each number.
        # This dictionary stores: value -> last_index
        last_occurrence = {}
        for i in range(n):
            last_occurrence[nums[i]] = i

        # Step 2: Identify the "base segments".
        # A base segment is a contiguous subarray that is self-contained.
        # This means all numbers within it have their last occurrences
        # within that segment, and no number inside requires it to extend further.
        
        num_segments = 0
        
        # `max_reach` tracks the maximum last_occurrence index among numbers
        # encountered in the current processing segment.
        # Initialize to -1, as array indices are non-negative.
        max_reach = -1 

        for i in range(n):
            # Update `max_reach`: The current segment must at least extend
            # to cover the last occurrence of the current number `nums[i]`.
            max_reach = max(max_reach, last_occurrence[nums[i]])

            # If the current index `i` matches `max_reach`, it signifies that
            # all numbers from the beginning of the current potential segment
            # up to `i` have their last occurrences at or before `i`.
            # This marks the end of a complete, self-contained base segment.
            if i == max_reach:
                num_segments += 1
        
        # Step 3: Calculate the total number of good partitions.
        # If there are `K` (num_segments) base segments, say B_1, B_2, ..., B_K,
        # these segments are guaranteed to have disjoint sets of values.
        # Any good partition can be formed by grouping these base segments
        # contiguously. For example, (B_1), (B_2, B_3), (B_4) is valid.
        # This is equivalent to choosing where to place cuts between these K segments.
        # There are (K - 1) potential cut points between the K segments.
        # At each point, we can either make a cut or not.
        # So, the total number of ways to partition these K segments is 2^(K-1).
        
        # Since `nums.length >= 1`, `num_segments` will always be at least 1.
        # If `num_segments` is 1, then `num_segments - 1` is 0, and 2^0 = 1.
        return pow(2, num_segments - 1, MOD)