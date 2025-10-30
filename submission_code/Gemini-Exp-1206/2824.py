class Solution:
    def isFascinating(self, n: int) -> bool:
        s = str(n) + str(2 * n) + str(3 * n)
        if len(s) != 9:
            return False
        
        digits = set()
        for char in s:
            if char == '0':
                return False
            digits.add(char)
        
        return len(digits) == 9