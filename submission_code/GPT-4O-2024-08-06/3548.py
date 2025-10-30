class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from itertools import permutations
        
        def is_palindrome(x):
            return str(x) == str(x)[::-1]
        
        def is_k_palindromic(x, k):
            return is_palindrome(x) and x % k == 0
        
        def is_good_integer(x, k):
            for perm in permutations(str(x)):
                if perm[0] != '0':  # Ensure no leading zero
                    num = int(''.join(perm))
                    if is_k_palindromic(num, k):
                        return True
            return False
        
        count = 0
        start = 10**(n-1)
        end = 10**n
        
        for i in range(start, end):
            if is_good_integer(i, k):
                count += 1
        
        return count