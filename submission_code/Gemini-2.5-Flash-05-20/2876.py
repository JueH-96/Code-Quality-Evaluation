from typing import List

class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        """
        Calculates the number of employees who worked at least target hours.

        Args:
            hours: A list of non-negative integers where hours[i] is the
                   hours worked by employee i.
            target: A non-negative integer representing the minimum hours
                    required.

        Returns:
            The integer denoting the number of employees who worked at least
            target hours.
        """
        # Initialize a counter for employees who meet the target.
        employees_met_target = 0

        # Iterate through each employee's hours.
        for h in hours:
            # Check if the current employee's hours are greater than or equal to the target.
            if h >= target:
                # If the condition is met, increment the counter.
                employees_met_target += 1
        
        # Return the final count of employees who met the target.
        return employees_met_target