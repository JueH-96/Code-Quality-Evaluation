from typing import List

class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        # Count and return the number of employees whose worked hours are at least the target
        return sum(1 for h in hours if h >= target)