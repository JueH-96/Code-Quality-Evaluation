class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # We can perform at most k operations.
        # After m operations, the final length = n - m.
        # Since m <= k, the minimum allowed final length is n - k.
        # We can view the operations as merging contiguous segments. 
        # Initially, each element is a segment. Merging segments never turns a 0 bit into a 1 
        # (because a merged value is the bitwise AND of the segments, and a 0 in any element keeps the bit 0).
        #
        # For any bit position b (0 <= b < 31), we ask: can we design a segmentation (by placing cuts)
        # of the array into contiguous segments such that in every segment, the bit b is eliminated (i.e.
        # at least one element in that segment has a 0 at bit b)? If so, then after merging appropriately,
        # the final values can all have 0 at bit b.
        #
        # Note the constraint: the final array length is n - m, where m <= k, so final length L must be 
        # at least n - k. In other words, we are allowed to have at most k merges, which means we can have
        # at minimum (n - k) segments.
        #
        # So for a bit b to be cleared, we must be able to partition the array into at least (n - k)
        # segments (by greedily cutting as early as possible) where each segment has at least one element 
        # with the b-th bit = 0.
        #
        # We can try for each bit from 0 to 30:
        #   - Parse through the array and whenever we see a segment (starting from the last cut index) that
        #     has an element with bit b = 0, we cut.
        #   - Count the number of segments obtained.
        #   - If the count is at least (n - k), then it is possible to clear bit b.
        #
        # Finally, the minimum possible OR is the OR of all bits which could NOT be cleared.
        #
        # Time complexity: O(31*n) which is acceptable.
        
        ans = 0
        # We'll process bit positions from 0 to 30.
        for bit in range(31):
            segment_count = 0
            # We want to greedily form segments.
            i = 0
            while i < n:
                # Try to form a segment [i, j] that contains at least one element with the current bit equal to 0.
                valid_segment = False
                # Expand this segment until we find an element with bit==0.
                while i < n and not valid_segment:
                    if (nums[i] >> bit) & 1 == 0:
                        valid_segment = True
                    i += 1
                if valid_segment:
                    segment_count += 1
                else:
                    # If we never find an element with bit==0, then break out.
                    break
            # To be able to clear this bit, we need to have formed at least (n - k) segments.
            # If not, then bit must remain in the final OR.
            if segment_count < (n - k):
                ans |= (1 << bit)
        return ans