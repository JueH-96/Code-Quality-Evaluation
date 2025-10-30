from math import gcd
from collections import Counter

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        first_digits = [int(str(num)[0]) for num in nums]
        last_digits = [num % 10 for num in nums]
        
        first_digits_count = Counter(first_digits)
        last_digits_count = Counter(last_digits)
        
        beautiful_pairs = 0
        
        for first_digit, first_count in first_digits_count.items():
            for last_digit, last_count in last_digits_count.items():
                if gcd(first_digit, last_digit) == 1:
                    beautiful_pairs += first_count * last_count
        
        # Subtract the pairs where the first and last digits are the same
        for num in nums:
            first_digit = int(str(num)[0])
            last_digit = num % 10
            if gcd(first_digit, last_digit) == 1:
                beautiful_pairs -= 1
        
        return beautiful_pairs // 2