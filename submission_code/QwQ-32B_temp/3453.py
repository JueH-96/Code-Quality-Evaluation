from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(s):
            if len(s) == n:
                result.append(s)
                return
            if len(s) == 0:
                backtrack('0')
                backtrack('1')
            else:
                last = s[-1]
                if last == '1':
                    backtrack(s + '0')
                    backtrack(s + '1')
                else:
                    # last is '0', so next must be '1'
                    backtrack(s + '1')
        
        backtrack('')
        return result