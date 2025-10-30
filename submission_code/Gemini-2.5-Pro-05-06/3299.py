import collections

class Solution:
    def maximumLength(self, nums: list[int]) -> int:
        freq = collections.Counter(nums)
        
        # Smallest possible length for any x is [x], length 1.
        # Example: nums = [2,3], max_overall_len = 1 from [2] or [3].
        # Since nums.length >= 2 (constraint), freq is not empty.
        max_overall_len = 1
        
        # Handle x = 1 case
        # Pattern: [1, 1, ..., 1]. Length must be odd (2m+1).
        # If freq[1] = c:
        #   - If c is odd, max length is c.
        #   - If c is even and positive, max length is c-1 (e.g., c=4 -> [1,1,1] length 3).
        count_of_1 = freq.get(1, 0)
        if count_of_1 > 0:
            if count_of_1 % 2 == 1:
                max_overall_len = max(max_overall_len, count_of_1)
            else:
                max_overall_len = max(max_overall_len, count_of_1 - 1)
        
        MAX_ELEMENT_VALUE = 10**9 # Max value for an element in nums from constraints.

        # Iterate through unique numbers in nums (excluding 1) as potential base x_cand
        for x_cand in freq: 
            if x_cand == 1:
                continue 
            
            # For x_cand to be the outermost element in a pattern of length >= 3,
            # we need at least two occurrences of x_cand.
            if freq.get(x_cand, 0) < 2:
                # Cannot form [x_cand, ..., x_cand]. Length 1 case [x_cand] is covered by init.
                continue

            # At this point, freq[x_cand] >= 2.
            # Try to form sequence for m=1: [x_cand, x_cand^2, x_cand]. Length 3.
            # Peak is x_cand^2 (x_cand^(2^1)), so m_idx=1.
            
            # Python handles large integers, direct multiplication is fine.
            # Check if squared value exceeds problem constraints for elements in nums.
            current_peak_candidate = x_cand * x_cand 
            
            if current_peak_candidate > MAX_ELEMENT_VALUE:
                # x_cand^2 is too large to be in nums.
                # Longest sequence with x_cand as outermost is just [x_cand] (length 1), already covered.
                continue

            if freq.get(current_peak_candidate, 0) >= 1:
                # Successfully formed [x_cand, x_cand^2, x_cand].
                # Current m_idx (exponent for k=2^m_idx) is 1. Length 2*1+1 = 3.
                max_overall_len = max(max_overall_len, 3)
                
                m_idx = 1
                # `element_forming_pair` is the term x_cand^(2^m_idx) which needs to appear >= 2 times.
                # Initially, this is x_cand^2 (which was current_peak_candidate).
                element_forming_pair = current_peak_candidate
                
                # Try to extend further:
                # [x_cand, ..., x_cand^(2^m_idx), (x_cand^(2^m_idx))^2, x_cand^(2^m_idx), ..., x_cand]
                while True:
                    # To extend, `element_forming_pair` (e.g. x_cand^2) must appear at least twice.
                    # The new peak `(element_forming_pair)^2` (e.g. x_cand^4) must appear at least once.
                    
                    # If element_forming_pair itself is already > MAX_ELEMENT_VALUE, it implies an issue
                    # or means x_cand was very large. freq.get will be 0. Loop will break.
                    # This check mainly for clarity/safety, as element_forming_pair was a previous valid peak.
                    if element_forming_pair > MAX_ELEMENT_VALUE : 
                        break

                    next_peak_candidate = element_forming_pair * element_forming_pair
                    if next_peak_candidate > MAX_ELEMENT_VALUE:
                        break # New peak is too large to be in nums.
                        
                    if freq.get(element_forming_pair, 0) >= 2 and freq.get(next_peak_candidate, 0) >= 1:
                        m_idx += 1 # Increment m_idx for the new, higher peak
                        max_overall_len = max(max_overall_len, 2 * m_idx + 1)
                        element_forming_pair = next_peak_candidate # This new peak becomes base for next pair check
                    else:
                        # Cannot extend (either element_forming_pair count is < 2, or new peak count is 0)
                        break
            # else: (freq.get(current_peak_candidate, 0) == 0)
                # x_cand^2 is not available. Cannot form length 3 sequence with x_cand as base.
                # Longest for this x_cand is [x_cand] (length 1), already covered.
                
        return max_overall_len