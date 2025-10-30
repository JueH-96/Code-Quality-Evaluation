from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        """
        Counts the number of passengers who are strictly older than 60.

        Args:
            details: A list of strings, where each string contains passenger information.
                     The format is: 10-digit phone number, 1-char gender, 2-digit age, 2-digit seat.

        Returns:
            The number of passengers with an age greater than 60.
        """
        
        # We can solve this concisely using a generator expression with the sum() function.
        # For each 'detail' string in the input list:
        # 1. We extract the age substring. The age is located at indices 11 and 12
        #    of the 15-character string. Slicing `detail[11:13]` gets these two characters.
        # 2. We convert this substring to an integer using int().
        # 3. We check if the resulting age is strictly greater than 60.
        # 4. The generator `(1 for detail in details if int(detail[11:13]) > 60)`
        #    yields a 1 for each passenger who meets the age criterion.
        # 5. The sum() function then totals these 1s to get the final count.
        
        return sum(1 for detail in details if int(detail[11:13]) > 60)