class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def is_special(n):
            divisors = []
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    divisors.append(i)
                    if i != n // i and i != 1:
                        divisors.append(n // i)
            return len(divisors) == 2
        
        special_count = 0
        for num in range(l, r + 1):
            if is_special(num):
                special_count += 1
        
        return (r - l + 1) - special_count