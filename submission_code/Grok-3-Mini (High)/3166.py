from typing import List
from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        counter = Counter(nums)
        freqs = sorted(counter.values())  # Sort frequencies in ascending order for potential short-circuiting
        min_f = freqs[0]
        for S in range(min_f, 0, -1):  # Iterate S from min_f down to 1
            if all((f + S) // (S + 1) <= f // S for f in freqs):  # Check if all frequencies can be partitioned with sizes S and S+1
                # If S works, compute the minimum number of groups
                G = 0
                for f in freqs:
                    G += (f + S) // (S + 1)  # Equivalent to ceil(f / (S+1))
                return G
        # The loop should always find a valid S (at least S=1 works), but if not, return 0 or handle error; this line should not be reached
        return 0