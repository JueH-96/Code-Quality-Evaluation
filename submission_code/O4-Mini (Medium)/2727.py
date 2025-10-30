from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        """
        Count how many passengers are strictly older than 60 years.
        Each detail string is formatted as:
          - chars 0-9:   phone number (ignored)
          - char  10:    gender (ignored)
          - chars 11-12: age in two digits
          - chars 13-14: seat number (ignored)
        """
        count = 0
        for info in details:
            age = int(info[11:13])
            if age > 60:
                count += 1
        return count