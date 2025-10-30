from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        """
        Count the number of passengers older than 60.
        Each string in details is structured as:
          - first 10 chars: phone number
          - 11th char: gender
          - 12th-13th chars: age
          - 14th-15th chars: seat number
        """
        count = 0
        for info in details:
            # Extract age substring and convert to integer
            age = int(info[11:13])
            if age > 60:
                count += 1
        return count