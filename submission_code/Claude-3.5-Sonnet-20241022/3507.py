class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def count_proper_divisors(n):
            count = 0
            i = 1
            while i * i <= n:
                if n % i == 0:
                    if i != n:
                        count += 1
                    if i != 1 and n//i != n:
                        count += 1
                i += 1
            return count
        
        special_count = 0
        for num in range(l, r+1):
            if count_proper_divisors(num) == 2:
                special_count += 1
                
        return (r - l + 1) - special_count