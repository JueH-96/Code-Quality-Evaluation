import math

class Solution:
    """
    This class provides a solution to distribute n candies among 3 children 
    such that no child gets more than limit candies.
    """
    def distributeCandies(self, n: int, limit: int) -> int:
        """
        Calculates the total number of ways to distribute n candies among 3 children
        with a limit on the maximum candies per child.

        Args:
            n: The total number of candies (positive integer).
            limit: The maximum number of candies any child can receive (positive integer).

        Returns:
            The total number of valid distribution ways.
            
        Constraints:
            1 <= n <= 50
            1 <= limit <= 50

        Method:
        We can iterate through all possible numbers of candies for the first two children
        and determine the required number for the third child. If the distribution
        satisfies the constraints for all three children, we count it as a valid way.

        Let c1, c2, c3 be the number of candies for child 1, child 2, and child 3 respectively.
        We need to find the number of integer solutions (c1, c2, c3) such that:
        1. c1 + c2 + c3 = n
        2. 0 <= c1 <= limit
        3. 0 <= c2 <= limit
        4. 0 <= c3 <= limit

        We can iterate through possible values of c1 (from 0 to limit) and c2 (from 0 to limit).
        For each pair (c1, c2), we calculate c3 = n - c1 - c2.
        We then check if this calculated c3 satisfies 0 <= c3 <= limit.
        If it does, we increment our count of valid distributions.
        """
        
        count = 0
        
        # Iterate through possible candies for child 1 (c1)
        # c1 can range from 0 up to the minimum of n and limit.
        # However, iterating up to limit is sufficient, as the check for c3
        # will handle cases where c1 or c1+c2 already exceed n.
        for c1 in range(limit + 1):
            # Iterate through possible candies for child 2 (c2)
            # c2 can range from 0 up to limit.
            for c2 in range(limit + 1):
                
                # Calculate the number of candies required for child 3 (c3)
                c3 = n - c1 - c2
                
                # Check if the distribution (c1, c2, c3) is valid:
                # 1. c1 <= limit (True due to loop range)
                # 2. c2 <= limit (True due to loop range)
                # 3. c3 >= 0   (Ensures total doesn't exceed n for c1 and c2)
                # 4. c3 <= limit (Ensures child 3 doesn't exceed the limit)
                
                # If c3 is within the valid range [0, limit], it's a valid distribution.
                if 0 <= c3 <= limit:
                    count += 1
                    
        # Return the total count of valid distributions found.
        return count