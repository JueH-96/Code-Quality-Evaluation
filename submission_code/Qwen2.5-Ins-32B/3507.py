class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def is_special(n):
            count = 0
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    count += 1
                    if i != n // i:
                        count += 1
            return count == 3
        
        count = 0
        for num in range(l, r + 1):
            if not is_special(num):
                count += 1
        return count