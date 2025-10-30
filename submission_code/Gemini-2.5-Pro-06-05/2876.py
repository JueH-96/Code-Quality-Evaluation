import collections

class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: list[int], target: int) -> int:
        """
        Calculates the number of employees who worked at least the target number of hours.

        Args:
            hours: A list of non-negative integers representing the hours worked by each employee.
            target: A non-negative integer representing the target hours.

        Returns:
            The number of employees who met the target.
        """
        
        # Initialize a counter for employees who met the target.
        count = 0
        
        # Iterate through the hours worked by each employee.
        for h in hours:
            # Check if the employee's hours are greater than or equal to the target.
            if h >= target:
                # If they met the target, increment the counter.
                count += 1
                
        return count

# A more concise, "Pythonic" way to write the same logic is using a generator expression with sum().
# This approach is generally preferred for its readability and conciseness.
#
# class Solution:
#     def numberOfEmployeesWhoMetTarget(self, hours: list[int], target: int) -> int:
#         # The generator expression (1 for h in hours if h >= target) yields a 1
#         # for each employee whose hours meet the target.
#         # The sum() function then adds up all the 1s, effectively counting
#         # the employees who met the condition.
#         return sum(1 for h in hours if h >= target)