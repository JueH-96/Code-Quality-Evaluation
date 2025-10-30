class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def count_proper_divisors(n):
            if n == 1:
                return 0
            count = 1  # 1 is always a proper divisor
            sqrt_n = int(n**0.5)
            for i in range(2, sqrt_n + 1):
                if n % i == 0:
                    count += 1
                    if i != n // i:
                        count += 1
            return count
        
        non_special_count = 0
        for num in range(l, r + 1):
            if count_proper_divisors(num) != 2:
                non_special_count += 1
        
        return non_special_count