class Solution:
    def isFascinating(self, n: int) -> bool:
        
        s = str(n) + str(2 * n) + str(3 * n)
        if len(s) != 9:
            return False
        
        counts = [0] * 10
        for char in s:
            digit = int(char)
            if digit == 0:
                return False
            counts[digit] += 1
        
        for i in range(1, 10):
            if counts[i] != 1:
                return False
        
        return True