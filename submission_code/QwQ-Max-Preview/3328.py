import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        s = math.ceil(2 * math.sqrt(k) - 2)
        while (s + 2) ** 2 < 4 * k:
            s += 1
        return s