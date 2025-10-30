class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def count_divisors(n):
            count = 0
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    if n / i == i:
                        count += 1
                    else:
                        count += 2
            return count

        count = 0
        for i in range(l, r + 1):
            if count_divisors(i) != 2:
                count += 1
        return count