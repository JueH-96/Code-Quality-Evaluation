from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(s):
            if len(s) == n:
                result.append(s)
                return
            if not s:
                # First character can be '0' or '1'
                backtrack('0')
                backtrack('1')
            else:
                last_char = s[-1]
                if last_char == '0':
                    backtrack(s + '1')
                else:
                    backtrack(s + '0')
                    backtrack(s + '1')
        
        backtrack('')
        return result