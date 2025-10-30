from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        def generate(n, s):
            if n == 0:
                return [s]
            if len(s) == 0 or s[-1] == '0':
                return generate(n - 1, s + '0') + generate(n - 1, s + '1')
            else:
                return generate(n - 1, s + '0') + generate(n - 1, s + '1')

        return generate(n, '')