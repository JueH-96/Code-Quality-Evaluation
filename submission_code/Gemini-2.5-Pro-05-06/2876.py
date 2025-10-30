from typing import List

class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        """
        Calculates the number of employees who worked at least 'target' hours.

        Args:
            hours: A list of non-negative integers where hours[i] is the hours
                   worked by employee i.
            target: A non-negative integer representing the minimum required
                    hours.

        Returns:
            The number of employees who met the target.
        """
        
        # Using a generator expression with sum().
        # For each employee's hours `h` in the `hours` list, `h >= target`
        # evaluates to True (1) if the employee met the target, and False (0) otherwise.
        # Summing these boolean values gives the total count of employees meeting the target.
        return sum(h >= target for h in hours)