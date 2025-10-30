from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        res = []
        
        def backtrack(path):
            if len(path) == n:
                res.append("".join(path))
                return
            # If path is empty or last char is '1', you can choose either '0' or '1'
            if not path or path[-1] == '1':
                # add '0'
                path.append('0')
                backtrack(path)
                path.pop()
                # add '1'
                path.append('1')
                backtrack(path)
                path.pop()
            else:
                # Last char is '0', can't add another '0' because that would form "00".
                path.append('1')
                backtrack(path)
                path.pop()
        
        backtrack([])
        return res