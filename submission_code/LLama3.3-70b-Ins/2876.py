from typing import List

class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        # Initialize a counter to store the number of employees who met the target
        count = 0
        
        # Iterate over each employee's working hours
        for hour in hours:
            # Check if the employee's working hours meet the target
            if hour >= target:
                # If the target is met, increment the counter
                count += 1
        
        # Return the total number of employees who met the target
        return count