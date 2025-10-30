class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from itertools import permutations
        
        def is_palindrome(s):
            return s == s[::-1]
        
        def is_divisible_by_k(num, k):
            return num % k == 0
        
        # Generate all possible numbers with n digits
        if n == 1:
            # Single digit numbers from 1 to 9
            possible_numbers = [str(i) for i in range(1, 10)]
        else:
            # Numbers from 10^(n-1) to 10^n - 1
            possible_numbers = [str(i) for i in range(10**(n-1), 10**n)]
        
        good_count = 0
        
        # Check each number if it can be rearranged to form a k-palindromic number
        for number in possible_numbers:
            # Generate all unique permutations of the number's digits
            perms = set([''.join(p) for p in permutations(number) if p[0] != '0'])
            # Check each permutation
            for perm in perms:
                num = int(perm)
                if is_palindrome(perm) and is_divisible_by_k(num, k):
                    good_count += 1
                    break
        
        return good_count