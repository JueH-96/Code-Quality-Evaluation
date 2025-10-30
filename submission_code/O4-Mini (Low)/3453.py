from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        # Generate all binary strings of length n without two consecutive '0's.
        res = []
        
        def dfs(pos: int, prev_zero: bool, path: List[str]):
            if pos == n:
                res.append("".join(path))
                return
            # Try placing '1'
            path.append('1')
            dfs(pos + 1, False, path)
            path.pop()
            # Try placing '0' if previous wasn't '0'
            if not prev_zero:
                path.append('0')
                dfs(pos + 1, True, path)
                path.pop()
        
        dfs(0, False, [])
        return res