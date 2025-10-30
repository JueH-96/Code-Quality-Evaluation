from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current):
            if len(current) == n:
                result.append(current)
                return
            if not current:
                # First character can be '0' or '1'
                backtrack(current + '0')
                backtrack(current + '1')
            else:
                last_char = current[-1]
                if last_char == '0':
                    # Next character must be '1'
                    backtrack(current + '1')
                else:
                    # Next character can be '0' or '1'
                    backtrack(current + '0')
                    backtrack(current + '1')
        
        backtrack('')
        return result