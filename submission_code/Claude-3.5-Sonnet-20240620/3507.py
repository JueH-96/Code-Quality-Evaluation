class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def count_proper_divisors(n):
            count = 0
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    if i * i == n:
                        count += 1
                    else:
                        count += 2
            return count - 1  # Subtract 1 to exclude n itself

        non_special_count = 0
        for num in range(l, r + 1):
            if count_proper_divisors(num) != 2:
                non_special_count += 1
        
        return non_special_count