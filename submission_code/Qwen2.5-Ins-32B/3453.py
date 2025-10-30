from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        def backtrack(length, path):
            if length == n:
                result.append(''.join(path))
                return
            if not path or path[-1] == '0':
                backtrack(length + 1, path + ['1'])
            else:
                backtrack(length + 1, path + ['0'])
                backtrack(length + 1, path + ['1'])
        
        result = []
        backtrack(0, [])
        if n > 1:
            backtrack(0, ['0'])
        return result