from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        # prev_or_values will store all distinct OR results of subarrays ending at the previous position
        prev_or_values = set()
        best_diff = float('inf')
        
        for x in nums:
            # Start new subarray at current element
            curr_or_values = {x}
            # Extend all previous subarrays by OR-ing with x
            for prev_val in prev_or_values:
                curr_or_values.add(prev_val | x)
            
            # Update best difference using all OR results ending here
            for or_val in curr_or_values:
                diff = abs(k - or_val)
                if diff < best_diff:
                    best_diff = diff
                    # Early exit if perfect match
                    if best_diff == 0:
                        return 0
            
            # Move to next index
            prev_or_values = curr_or_values
        
        return best_diff