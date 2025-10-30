class Solution:
    def validStrings(self, n: int) -> List[str]:
        from typing import List
        
        def dfs(prefix: str, length: int, result: List[str]) -> None:
            if len(prefix) == length:
                result.append(prefix)
                return
            # If empty or last character was '1', we can add either '0' or '1'
            if not prefix or prefix[-1] == '1':
                dfs(prefix + '0', length, result)
                dfs(prefix + '1', length, result)
            else:
                # If last character was '0', next must be '1'
                dfs(prefix + '1', length, result)
        
        ans = []
        dfs("", n, ans)
        return ans