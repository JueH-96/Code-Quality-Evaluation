from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current: str, last_char: str) -> None:
            if len(current) == n:
                result.append(current)
                return
            if last_char == '0':
                # Only allowed to add '1' after '0'
                backtrack(current + '1', '1')
            elif last_char == '1':
                # Can add either '0' or '1' after '1'
                backtrack(current + '0', '0')
                backtrack(current + '1', '1')
            else:
                # Initial call: no previous character, start with '0' and '1'
                backtrack(current + '0', '0')
                backtrack(current + '1', '1')
        
        backtrack('', None)
        return result