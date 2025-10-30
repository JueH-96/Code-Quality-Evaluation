from itertools import permutations

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        
        def is_palindrome(x):
            return str(x) == str(x)[::-1]
        
        def is_k_palindromic(x, k):
            return is_palindrome(x) and x % k == 0
        
        def generate_numbers(n):
            if n == 1:
                return [str(i) for i in range(1, 10)]
            numbers = []
            for perm in permutations('1234567890', n):
                if perm[0] != '0':
                    numbers.append(''.join(perm))
            return numbers
        
        def count_good_integers(n, k):
            count = 0
            numbers = generate_numbers(n)
            for num in numbers:
                for perm in set(permutations(num)):
                    perm_num = int(''.join(perm))
                    if is_k_palindromic(perm_num, k):
                        count += 1
                        break
            return count
        
        return count_good_integers(n, k)