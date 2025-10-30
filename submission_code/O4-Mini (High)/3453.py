from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        # Initialize for n = 1
        res = ['0', '1']
        # Build strings of length 2..n
        for _ in range(2, n + 1):
            new_res = []
            for s in res:
                # Always can append '1'
                new_res.append(s + '1')
                # Append '0' only if last char was '1' (to avoid "00")
                if s[-1] == '1':
                    new_res.append(s + '0')
            res = new_res
        return res