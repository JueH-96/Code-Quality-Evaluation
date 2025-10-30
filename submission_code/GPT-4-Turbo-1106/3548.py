from itertools import permutations

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        def is_palindrome(num):
            return str(num) == str(num)[::-1]
        
        count = 0
        # Generate all possible numbers with n digits
        for num in range(10**(n-1), 10**n):
            if num % k == 0 and is_palindrome(num):
                count += 1
            else:
                # Check if the digits can be rearranged to form a k-palindromic integer
                for perm in set(permutations(str(num))):
                    if perm[0] != '0':  # Avoid leading zeros
                        perm_num = int(''.join(perm))
                        if perm_num % k == 0 and is_palindrome(perm_num):
                            count += 1
                            break
        return count