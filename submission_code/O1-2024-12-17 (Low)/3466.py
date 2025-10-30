class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        """
        We want to count the number of subarrays whose bitwise AND equals k.
        
        Observations/Approach:
        1. For a subarray to have AND == k, every element in that subarray must contain
           all bits that k has. In other words, if an element e is in that subarray,
           then (e & k) == k. If that is not true for some element, that element
           immediately "breaks" any subarray containing it from having AND == k.
           Therefore, we can split the array into contiguous "segments" separated
           by these "bad" elements (those that are missing at least one bit from k).

        2. Within each such "good" segment (where all elements e satisfy (e & k) == k),
           we then must ensure that the subarray does not keep *extra* bits that
           are not in k. If there is a bit outside of k that is present in every
           element of the subarray, that subarray's AND would include that bit,
           making the AND larger than k. Thus, for the AND to be exactly k, every
           bit not in k must be "cleared" by at least one element in the subarray.

        3. Implementation detail for each good segment:
           - Let extra[i] = nums[i] & ~k  (the bits of nums[i] that are not in k).
           - We want sub-subarrays where the AND of all extra[i] in that subarray is 0.
             Equivalently, for each bit not in k, there must be at least one element
             in the sub-subarray that does not have that bit. 
           - We track for each "outside" bit b (where (k >> b) & 1 == 0) the last index
             where that bit was "cleared" (i.e., an element that does NOT have bit b).
             Let lastpos[b] be that index.
           - As we iterate through the segment with index i, for each outside bit b:
                if ((extra[i] >> b) & 1) == 0, then we update lastpos[b] = i
             The sub-subarray up to i is valid only if for all b, lastpos[b] >= start_of_segment.
             In practice, we take the minimum of lastpos[b], call it minLP. 
             All valid sub-subarrays ending at i can start anywhere in [segment_start..minLP].
             Therefore, the new valid sub-subarrays ending at i is (minLP - segment_start + 1)
             (assuming minLP >= segment_start).
             
           - Edge case: If k has all 32 bits that appear in the numbers, then "outside" bits set is empty.
             In that scenario, every sub-subarray in the good segment is valid, giving length*(length+1) // 2 sub-subarrays
             for a segment of length "length".

           - Another edge case: If in a good segment there is some outside bit b that appears in every element
             (no element ever clears it), then no sub-subarray can drop that bit from the AND, so the count
             is zero in that segment.

        4. We sum the valid counts for all segments and return the total.
        """
        import sys
        
        n = len(nums)
        # Identify bits (up to 31 for constraints 0..10^9) we need to worry about.
        # bits_of_interest are those bits not in k (those that must be cleared by at least one element).
        bits_of_interest = []
        for b in range(32):
            if ((k >> b) & 1) == 0:
                bits_of_interest.append(b)
        
        # A helper to check "bad" elements:
        # A "bad" element is one that does not contain all bits of k,
        # i.e. (elem & k) != k
        def is_bad(elem: int) -> bool:
            return (elem & k) != k
        
        total_count = 0
        
        start = 0
        while start < n:
            # Skip any "bad" element to find the start of the next good segment
            if is_bad(nums[start]):
                start += 1
                continue
            
            # We have start pointing to a good element, now find the segment_end
            segment_start = start
            end = start
            while end < n and not is_bad(nums[end]):
                end += 1
            # Now [segment_start..end-1] is the "good" segment (inclusive)

            segment_length = end - segment_start
            
            # If there are no bits_of_interest, this means k's bits are the only
            # ones that might appear, so the entire sub-segment is valid.
            # The number of sub-subarrays in a segment of length L is L*(L+1)//2.
            if not bits_of_interest:
                total_count += segment_length * (segment_length + 1) // 2
                start = end
                continue
            
            # Next, we must ensure that each bit not in k is cleared at least once
            # in every valid sub-subarray. We'll track the last position that cleared
            # (did not have) each outside bit.
            lastpos = {b: -1 for b in bits_of_interest}
            
            # First, check if for some outside bit b, every element has that bit.
            # If so, the sub-subarray AND can never drop that bit, so count=0 for the segment.
            can_form_valid = True
            for b in bits_of_interest:
                # If no element in [segment_start..end-1] ever clears bit b,
                # then it's impossible to form a sub-subarray that ANDs to k.
                # We'll check that at least one element in the segment does NOT have bit b.
                found_clear = False
                for i in range(segment_start, end):
                    if ((nums[i] & ~k) >> b) & 1 == 0:
                        found_clear = True
                        break
                if not found_clear:
                    can_form_valid = False
                    break
            
            if not can_form_valid:
                # No valid sub-subarrays in this segment; skip
                start = end
                continue
            
            # Otherwise, do the counting with the "lastpos" approach
            seg_count = 0
            minLP = -1  # will track the minimum of lastpos[b]
            
            # We iterate i over [segment_start..end-1]
            for i in range(segment_start, end):
                current_extra = nums[i] & ~k
                # Update lastpos for each bit of interest if it's cleared by nums[i]
                # i.e., if bit b is NOT set in current_extra, that means it "clears" b
                for b in bits_of_interest:
                    if ((current_extra >> b) & 1) == 0:
                        lastpos[b] = i
                # Compute minLP => minimum lastpos[b] among all bits of interest
                # If minLP < segment_start at any point or we have -1 for some bit, 
                # that means we haven't yet cleared all bits in [segment_start..i].
                current_min_lastpos = min(lastpos[b] for b in bits_of_interest)
                
                if current_min_lastpos < segment_start:
                    # We haven't cleared all bits up to i
                    # So, no new valid sub-subarrays ending at i
                    continue
                else:
                    # All bits_of_interest are cleared by at least one element
                    # in the sub-subarray that starts from segment_start..current_min_lastpos
                    # The valid starts are from segment_start..current_min_lastpos inclusive
                    seg_count += (current_min_lastpos - segment_start + 1)
            
            total_count += seg_count

            start = end  # move to next segment
        
        return total_count