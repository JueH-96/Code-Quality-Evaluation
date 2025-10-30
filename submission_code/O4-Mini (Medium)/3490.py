from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count of even and odd numbers for the "constant parity" case (c = 0).
        count_even = 0
        count_odd = 0
        
        # Count of runs of alternating parity for the "alternating parity" case (c = 1).
        runs = 0
        prev_parity = None
        
        for x in nums:
            p = x & 1
            if p == 0:
                count_even += 1
            else:
                count_odd += 1
            
            # Whenever parity changes (or at the first element), we start a new run.
            if prev_parity is None or p != prev_parity:
                runs += 1
                prev_parity = p
        
        # The answer is the best of:
        # 1) longest subsequence with all adjacent sums even => pick all evens or all odds
        # 2) longest alternating-parity subsequence => one element per run
        return max(count_even, count_odd, runs)