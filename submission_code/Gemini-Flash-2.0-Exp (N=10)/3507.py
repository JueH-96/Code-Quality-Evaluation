import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def is_special(n):
            if n <= 3:
                return False
            count = 0
            for i in range(1, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    if n // i == i:
                        count += 1
                    else:
                        count += 2
            return count == 3

        special_count = 0
        for i in range(l, r + 1):
            if is_special(i):
                special_count += 1
        return (r - l + 1) - special_count