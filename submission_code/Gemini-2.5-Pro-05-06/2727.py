from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        """
        Counts the number of passengers strictly older than 60 years.

        Args:
            details: A list of strings, where each string provides passenger information.
                     The format of each string (length 15) is:
                     - Phone number (10 chars, indices 0-9)
                     - Gender (1 char, index 10)
                     - Age (2 chars, indices 11-12)
                     - Seat (2 chars, indices 13-14)

        Returns:
            The number of passengers strictly older than 60.
        """
        
        # Using a generator expression with sum() for a concise solution.
        # For each 'detail_str' in the 'details' list:
        # 1. Extract the age part: detail_str[11:13] (characters at index 11 and 12).
        # 2. Convert this age string to an integer: int(...).
        # 3. Check if the age is strictly greater than 60.
        # 4. If true, this passenger contributes 1 to the sum.
        return sum(1 for detail_str in details if int(detail_str[11:13]) > 60)