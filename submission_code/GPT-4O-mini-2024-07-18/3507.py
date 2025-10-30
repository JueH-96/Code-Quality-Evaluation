class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def is_special(n):
            if n < 2:
                return False
            count = 0
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    if i != n:
                        count += 1
                    if i != 1 and i * i != n and (n // i) != n:
                        count += 1
                if count > 2:
                    return False
            return count == 2
        
        special_count = 0
        for num in range(l, r + 1):
            if is_special(num):
                special_count += 1
        
        return (r - l + 1) - special_count