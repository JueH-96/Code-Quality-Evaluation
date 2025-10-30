class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from collections import Counter
        
        def is_k_palindromic(x, k):
            s = str(x)
            return s == s[::-1] and x % k == 0
        
        def can_form_k_palindrome(counter, n, k):
            odd_count = sum(1 for count in counter.values() if count % 2 != 0)
            if n % 2 == 0:
                return odd_count == 0
            else:
                return odd_count <= 1
        
        def count_good_integers(n, k):
            if n == 1:
                return sum(1 for i in range(1, 10) if i % k == 0)
            
            count = 0
            for first_digit in range(1, 10):
                for last_digit in range(10):
                    if n % 2 == 0 and first_digit != last_digit:
                        continue
                    remaining_length = n - 2 if n > 1 else 0
                    for middle in range(10 ** remaining_length):
                        num_str = str(first_digit) + str(middle).zfill(remaining_length) + str(last_digit)
                        num = int(num_str)
                        if num % k == 0:
                            count += 1
            return count
        
        return count_good_integers(n, k)