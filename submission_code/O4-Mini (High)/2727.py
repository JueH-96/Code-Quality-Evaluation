from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        count = 0
        for info in details:
            # Age is at positions 11 and 12 (0-based indexing)
            age = int(info[11:13])
            if age > 60:
                count += 1
        return count