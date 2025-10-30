from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        res = []
        
        def helper(s: str, last: str) -> None:
            if len(s) == n:
                res.append(s)
                return
            if last is None:
                helper(s + '0', '0')
                helper(s + '1', '1')
            else:
                if last == '0':
                    helper(s + '1', '1')
                else:
                    helper(s + '0', '0')
                    helper(s + '1', '1')
        
        helper("", None)
        return res