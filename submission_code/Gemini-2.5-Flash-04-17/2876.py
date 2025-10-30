from typing import List

class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        """
        Counts the number of employees whose worked hours meet or exceed the target.

        Args:
            hours: A list of integers representing hours worked by each employee.
            target: An integer representing the minimum required hours.

        Returns:
            The number of employees who worked at least target hours.
        """
        # Initialize a counter for employees who met the target
        count = 0
        
        # Iterate through the list of hours
        for hour in hours:
            # Check if the current employee's hours meet or exceed the target
            if hour >= target:
                # If yes, increment the counter
                count += 1
                
        # Return the final count
        return count