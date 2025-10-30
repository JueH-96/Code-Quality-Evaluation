class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def is_special(n):
            divisors = []
            for i in range(1, n):
                if n % i == 0:
                    divisors.append(i)
            return len(divisors) == 2

        count = 0
        special_numbers = set()
        
        limit = int(r**0.5) + 1
        
        for i in range(2, limit):
            special_numbers.add(i*i)
        
        special_count = 0
        for i in special_numbers:
            if l <= i <= r:
                special_count += 1
        
        return (r - l + 1) - special_count