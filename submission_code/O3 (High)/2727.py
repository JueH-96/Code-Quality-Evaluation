from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        """
        Count passengers whose age (characters at index 11 and 12 in each detail string)
        is strictly greater than 60.
        """
        seniors = 0
        for info in details:
            # Extract age substring: characters at positions 11 and 12
            age = int(info[11:13])
            if age > 60:
                seniors += 1
        return seniors