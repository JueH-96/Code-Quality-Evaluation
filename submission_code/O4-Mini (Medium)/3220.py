from typing import List

class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        tested = 0
        # decrements counts how many times we've tested a device so far,
        # which equals how many times later batteries have been reduced by 1.
        decrements = 0
        
        for bp in batteryPercentages:
            # effective battery after prior reductions
            if bp - decrements > 0:
                tested += 1
                decrements += 1
        
        return tested