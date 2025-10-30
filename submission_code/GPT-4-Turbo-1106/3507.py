class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def is_special(n):
            if n < 4:
                return False
            sqrt_n = int(n ** 0.5)
            if sqrt_n * sqrt_n != n:
                return False
            for i in range(2, int(sqrt_n ** 0.5) + 1):
                if sqrt_n % i == 0:
                    return False
            return True
        
        count = 0
        for x in range(l, r + 1):
            if not is_special(x):
                count += 1
        return count