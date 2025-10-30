from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        """
        Counts the number of passengers who are strictly more than 60 years old.

        Args:
            details: A 0-indexed array of strings, where each string is 15 characters long
                     and contains passenger information.
                     The age is represented by the characters at indices 11 and 12.

        Returns:
            The total number of passengers whose age is greater than 60.
        """
        senior_count = 0  # Initialize a counter for senior passengers

        # Iterate through each passenger's details string
        for passenger_info in details:
            # Extract the age part from the string.
            # According to the problem description, age is at indices 11 and 12.
            # In Python slicing, [start:end] includes 'start' but excludes 'end'.
            # So, to get characters at index 11 and 12, we use slice [11:13].
            age_str = passenger_info[11:13]

            # Convert the extracted age string to an integer
            age = int(age_str)

            # Check if the passenger's age is strictly greater than 60
            if age > 60:
                senior_count += 1  # Increment the counter if they are a senior

        return senior_count  # Return the final count of senior passengers