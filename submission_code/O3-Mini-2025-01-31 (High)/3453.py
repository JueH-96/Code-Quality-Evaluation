from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(cur: str):
            if len(cur) == n:
                result.append(cur)
                return
            # If the last character is '0', we cannot add another '0' to avoid "00"
            if cur and cur[-1] == '0':
                backtrack(cur + '1')
            else:
                # We can add either '0' or '1'
                backtrack(cur + '0')
                backtrack(cur + '1')
                
        backtrack("")  # start with an empty string
        return result