from typing import List

class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        """
        Count how many employees have worked at least 'target' hours.
        """
        count = 0
        for h in hours:
            if h >= target:
                count += 1
        return count