class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def is_special(x: int) -> bool:
            if x < 4:
                return False
            sqrt_x = int(x**0.5)
            if sqrt_x * sqrt_x == x:
                return sqrt_x == 2
            return False

        count = 0
        for num in range(l, r + 1):
            if not is_special(num):
                count += 1
        return count