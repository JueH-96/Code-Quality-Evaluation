import math
from collections import Counter

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        unique_digit_sets = set()
        m = (n + 1) // 2
        start = 10 ** (m - 1)
        end = 10 ** m
        
        for first_part_num in range(start, end):
            first_part_str = str(first_part_num)
            if n % 2 == 0:
                pal_str = first_part_str + first_part_str[::-1]
            else:
                pal_str = first_part_str + first_part_str[:-1][::-1]
            pal_num = int(pal_str)
            if pal_num % k == 0:
                digits_tuple = tuple(sorted(pal_str))
                unique_digit_sets.add(digits_tuple)
        
        total = 0
        for digits_tuple in unique_digit_sets:
            freq = Counter(digits_tuple)
            n_current = n
            numerator = math.factorial(n_current)
            denominator = 1
            for count in freq.values():
                denominator *= math.factorial(count)
            total_perm = numerator // denominator
            
            perm_with_zero = 0
            if '0' in freq:
                new_freq = freq.copy()
                new_freq['0'] -= 1
                if new_freq['0'] == 0:
                    del new_freq['0']
                numerator2 = math.factorial(n_current - 1)
                denominator2 = 1
                for count in new_freq.values():
                    denominator2 *= math.factorial(count)
                perm_with_zero = numerator2 // denominator2
            
            valid = total_perm - perm_with_zero
            total += valid
        
        return total