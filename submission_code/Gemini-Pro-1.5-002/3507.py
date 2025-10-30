class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        
        def is_special(n):
            count = 0
            for i in range(1, n):
                if n % i == 0:
                    count += 1
            return count == 2

        count = 0
        for i in range(l, r + 1):
            if not is_special(i):
                count += 1
        return count