class Solution:
    def isFascinating(self, n: int) -> bool:
        s = str(n) + str(2 * n) + str(3 * n)
        if len(s) != 9:
            return False
        
        seen = set()
        for digit in s:
            if not '1' <= digit <= '9':
                return False
            if digit in seen:
                return False
            seen.add(digit)
        return True