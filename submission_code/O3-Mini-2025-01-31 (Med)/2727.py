from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        count = 0
        for detail in details:
            # Extract age from the string (characters 11 and 12)
            age = int(detail[11:13])
            if age > 60:
                count += 1
        return count