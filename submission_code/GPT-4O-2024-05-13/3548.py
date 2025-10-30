class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from itertools import permutations
        
        def is_palindrome(x):
            return str(x) == str(x)[::-1]
        
        def generate_numbers(n):
            digits = '123456789'
            if n == 1:
                return [int(d) for d in digits]
            else:
                return [int(''.join(p)) for p in permutations(digits, n) if p[0] != '0']
        
        count = 0
        for num in generate_numbers(n):
            if num % k == 0:
                if any(is_palindrome(int(''.join(p))) for p in permutations(str(num))):
                    count += 1
        
        return count