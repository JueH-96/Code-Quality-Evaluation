from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        def backtrack(path):
            if len(path) == n:
                result.append(''.join(path))
                return
            if not path or path[-1] == '1':
                # Can add '0'
                path.append('0')
                backtrack(path)
                path.pop()
                # Can add '1'
                path.append('1')
                backtrack(path)
                path.pop()
            else:  # path[-1] == '0'
                # Can only add '1'
                path.append('1')
                backtrack(path)
                path.pop()
        backtrack([])
        return result