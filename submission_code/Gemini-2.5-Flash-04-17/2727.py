from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        """
        Counts the number of passengers strictly more than 60 years old.

        Args:
            details: A list of strings, where each string contains passenger
                     information in the format phone (10 chars), gender (1 char),
                     age (2 chars), seat (2 chars).

        Returns:
            The number of passengers strictly older than 60.
        """
        count = 0
        for detail in details:
            # The age information is located at indices 11 and 12
            # We extract the substring representing the age
            age_str = detail[11:13]
            
            # Convert the age substring to an integer
            age = int(age_str)
            
            # Check if the age is strictly greater than 60
            if age > 60:
                count += 1
                
        return count