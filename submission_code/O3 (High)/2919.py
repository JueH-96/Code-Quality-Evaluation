from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort so that we meet the smallest quotas first
        usageLimits.sort()
        
        groups = 0          # how many groups have been formed so far
        available = 0       # total still-unassigned occurrences we own
        
        for cnt in usageLimits:
            available += cnt                # add everything this number can still supply
            
            # The next group we want to create must have size groups+1.
            if available >= groups + 1:     # if we can afford that many DISTINCT numbers
                groups += 1                 # we form the new group
                available -= groups         # consume one occurrence from `groups` distinct numbers
        
        return groups