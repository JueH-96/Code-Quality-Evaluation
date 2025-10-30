from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current):
            if len(current) == n:
                result.append(''.join(current))
                return
            if not current:
                backtrack(['0'])
                backtrack(['1'])
            else:
                if current[-1] == '0':
                    backtrack(current + ['1'])
                elif current[-1] == '1':
                    backtrack(current + ['0'])
                    backtrack(current + ['1'])
        
        backtrack([])
        return result