class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def count_special(n):
            count = 0
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    count += 1
                    if i != n // i:
                        count += 1
            return count == 1
        
        return sum(1 for x in range(l, r + 1) if not count_special(x))