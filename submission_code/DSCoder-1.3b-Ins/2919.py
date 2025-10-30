import math
from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        groups = 1
        usage = 0
        for limit in usageLimits:
            usage += limit
            if usage >= groups:
                groups += 1
                usage -= groups
        return groups