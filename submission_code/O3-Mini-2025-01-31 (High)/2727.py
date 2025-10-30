from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        count = 0
        for detail in details:
            # Extract the age from index 11 to 13 (as the age is specified in two characters)
            age = int(detail[11:13])
            if age > 60:
                count += 1
        return count