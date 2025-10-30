from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        res = []
        def backtrack(s):
            if len(s) == n:
                res.append(s)
                return
            # Always can add '1'
            backtrack(s + '1')
            # Can add '0' only if the last character is not '0'
            if not s or s[-1] != '0':
                backtrack(s + '0')
        backtrack('')
        return res