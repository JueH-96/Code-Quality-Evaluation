from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 0:
            return []
        current = ['0', '1']
        if n == 1:
            return current
        
        for _ in range(1, n):
            next_level = []
            for s in current:
                if s[-1] == '0':
                    next_level.append(s + '1')
                else:
                    next_level.append(s + '0')
                    next_level.append(s + '1')
            current = next_level
        
        return current