from typing import List

class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        """
        Counts how many employees have worked at least `target` hours.

        Parameters
        ----------
        hours : List[int]
            A list where hours[i] is the number of hours the i-th employee has worked.
        target : int
            The minimum required number of hours.

        Returns
        -------
        int
            The number of employees who met or exceeded the target hours.
        """
        # Iterate through the list and count entries meeting or exceeding the target.
        return sum(h >= target for h in hours)