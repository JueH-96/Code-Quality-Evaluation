import collections
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        counts = collections.Counter(nums)
        max_overall_length = 1

        # Handle the special case where x = 1.
        # If x=1, the pattern is [1, 1, ..., 1]. Its length is simply the count of 1s available.
        if 1 in counts:
            max_overall_length = counts[1]

        # Maximum value for nums[i] is 10^9.
        # If current_val > SQRT_MAX_VAL, then current_val * current_val will exceed 10^9.
        # sqrt(10^9) is approximately 31622.77.
        SQRT_MAX_VAL = 31622 

        # Iterate through unique numbers as potential starting points 'x'.
        # Sorting keys is not strictly necessary for correctness but ensures consistent processing order.
        for start_val in sorted(counts.keys()):
            if start_val == 1:
                continue # Already handled

            # Build the potential chain of powers: x, x^2, x^4, ..., x^(2^k)
            # Store values in path_vals.
            path_vals = []
            current_val = start_val
            
            while current_val <= 10**9 and current_val in counts:
                path_vals.append(current_val)
                
                # Check if the next square would exceed the maximum allowed value (10^9).
                # This prevents unnecessary calculations and potential overflow.
                if current_val > SQRT_MAX_VAL:
                    break 
                
                current_val *= current_val
            
            # Now, path_vals contains [x, x^2, ..., x^(2^k)] for the current start_val.
            # 'k' here is len(path_vals) - 1.
            
            # We need to find the largest 'm' such that the pattern [x, ..., x^(2^m), ..., x] can be formed.
            # The length of this pattern is 2*m + 1.
            # The pattern requires:
            # - counts[x^(2^i)] >= 2 for i from 0 to m-1 (intermediate elements)
            # - counts[x^(2^m)] >= 1 (peak element)
            
            current_m_for_this_start_val = 0 # Default to m=0, length 1 ([x] pattern)
            
            # Iterate m_idx from k (max possible m) down to 0.
            # m_idx represents the 'm' in x^(2^m), which is the peak of the pattern.
            for m_idx in range(len(path_vals) - 1, -1, -1):
                can_form_m_pattern = True
                
                # Check counts for intermediate elements (x^(2^i) for i < m_idx).
                # These elements need at least 2 copies.
                for i in range(m_idx):
                    if counts[path_vals[i]] < 2:
                        can_form_m_pattern = False
                        break
                
                if not can_form_m_pattern:
                    continue # Cannot form pattern with this m_idx, try a smaller m_idx

                # Check count for the peak element (x^(2^m_idx)).
                # This element needs at least 1 copy.
                if counts[path_vals[m_idx]] < 1:
                    can_form_m_pattern = False
                
                if can_form_m_pattern:
                    # Found the largest valid 'm_idx' for this start_val.
                    # The length of this pattern is 2*m_idx + 1.
                    current_m_for_this_start_val = m_idx
                    break 
            
            # Update the maximum overall length found so far.
            max_overall_length = max(max_overall_length, 2 * current_m_for_this_start_val + 1)
            
        return max_overall_length