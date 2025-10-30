import collections # Although not strictly needed for this simple solution,
                 # it's good practice to be aware of standard libraries.
                 # Specifically for type hinting 'List' below.
from typing import List # Import List for type hinting

class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        """
        Counts the number of employees who worked at least 'target' hours.

        Args:
            hours: A list of non-negative integers representing hours worked 
                   by each employee. The list is 0-indexed, and its length is n.
            target: A non-negative integer representing the minimum required hours.

        Returns:
            An integer denoting the number of employees who worked at least 
            target hours.
        """
        
        # Initialize a counter for employees who met the target
        count = 0
        
        # Iterate through the hours worked by each employee
        for h in hours:
            # Check if the employee's hours are greater than or equal to the target
            if h >= target:
                # If they met the target, increment the counter
                count += 1
                
        # Return the final count
        return count

# Example usage (for testing purposes, not part of the required class structure):
# sol = Solution()
# hours1 = [0, 1, 2, 3, 4]
# target1 = 2
# print(f"Example 1 Input: hours = {hours1}, target = {target1}")
# print(f"Example 1 Output: {sol.numberOfEmployeesWhoMetTarget(hours1, target1)}") # Expected: 3

# hours2 = [5, 1, 4, 2, 2]
# target2 = 6
# print(f"Example 2 Input: hours = {hours2}, target = {target2}")
# print(f"Example 2 Output: {sol.numberOfEmployeesWhoMetTarget(hours2, target2)}") # Expected: 0

# Alternative concise Pythonic solution using list comprehension/generator expression and sum:
# class Solution:
#     def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
#         # sum() treats True as 1 and False as 0
#         return sum(h >= target for h in hours)