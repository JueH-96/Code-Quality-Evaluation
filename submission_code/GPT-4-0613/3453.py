from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        elif n == 2:
            return ["01", "10", "11"]
        else:
            result = []
            for s in self.validStrings(n - 1):
                result.append(s + "1")
                if s[-1] == "1":
                    result.append(s + "0")
            return result