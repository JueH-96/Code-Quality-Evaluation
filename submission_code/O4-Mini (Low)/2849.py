from bisect import bisect_left
from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        
        # For each starting index i, we build subarrays [i..j]
        for i in range(n):
            # maintain a sorted multiset of the elements in the current subarray
            s = []
            imbalance = 0
            
            for j in range(i, n):
                x = nums[j]
                # Find insertion position in the current sorted list s
                pos = bisect_left(s, x)
                # Determine neighbors in the old list before insertion
                L_old = s[pos-1] if pos > 0 else None
                R_old = s[pos]   if pos < len(s) else None
                
                # If both old neighbors exist and had a gap >1, that gap is about to be broken
                old_gap = 1 if (L_old is not None and R_old is not None and R_old - L_old > 1) else 0
                
                # Insert x into the sorted list
                s.insert(pos, x)
                
                # Determine new neighbors after insertion
                L_new = s[pos-1] if pos > 0 else None
                R_new = s[pos+1] if pos+1 < len(s) else None
                
                # New possible gaps
                new_gap_left  = 1 if (L_new is not None and x - L_new > 1) else 0
                new_gap_right = 1 if (R_new is not None and R_new - x > 1) else 0
                
                # Update imbalance for this subarray
                imbalance += new_gap_left + new_gap_right - old_gap
                
                # Accumulate into total
                total += imbalance
        
        return total