from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        return len([s for s in details if int(s[11:13]) > 60])