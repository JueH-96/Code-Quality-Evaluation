from itertools import product
from collections import Counter

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        def is_palindrome(x):
            s = str(x)
            return s == s[::-1]
        
        def can_form_palindrome(digits):
            cnt = Counter(digits)
            odd = 0
            for v in cnt.values():
                if v % 2 != 0:
                    odd += 1
            return odd <= 1
        
        def generate_numbers(n):
            if n == 1:
                return [str(i) for i in range(1, 10)]
            else:
                return [''.join(map(str, num)) for num in product(range(0, 10), repeat=n) if num[0] != 0]
        
        count = 0
        for num_str in generate_numbers(n):
            digits = list(num_str)
            if can_form_palindrome(digits):
                num = int(num_str)
                if num % k == 0:
                    count += 1
        return count