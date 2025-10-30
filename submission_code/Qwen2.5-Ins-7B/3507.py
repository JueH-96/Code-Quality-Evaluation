class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def isSpecial(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True if len([i for i in range(1, n) if n % i == 0]) == 2 else False
        
        return sum(not isSpecial(x) for x in range(l, r + 1))