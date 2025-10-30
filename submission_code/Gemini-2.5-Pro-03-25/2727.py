import collections
from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        """
        Counts the number of passengers strictly older than 60 based on the details strings.

        Args:
            details: A list of strings, where each string contains passenger information.
                     Format: 10 digits (phone) + 1 char (gender) + 2 digits (age) + 2 digits (seat).

        Returns:
            The number of passengers whose age is strictly greater than 60.
        """
        senior_count = 0  # Initialize a counter for seniors

        # Iterate through each passenger's detail string in the list
        for detail_string in details:
            # The age is represented by the characters at indices 11 and 12.
            # Extract the age substring.
            # Example: "7868190130M7522" -> age substring is "75"
            age_str = detail_string[11:13] 
            
            # Convert the age substring (which is a string) into an integer.
            age_int = int(age_str)
            
            # Check if the passenger's age is strictly greater than 60.
            if age_int > 60:
                # If the condition is met, increment the senior counter.
                senior_count += 1
                
        # After checking all passengers, return the total count of seniors.
        return senior_count