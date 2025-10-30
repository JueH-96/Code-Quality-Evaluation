class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        def is_palindrome(n):
            return str(n) == str(n)[::-1]

        def is_divisible(n, k):
            return n % k == 0

        def is_good(n, k):
            digits = [int(d) for d in str(n)]
            for i in range(10**(len(digits)-1), 10**len(digits)):
                if is_palindrome(i) and is_divisible(i, k):
                    if sorted([int(d) for d in str(i)]) == sorted(digits):
                        return True
            return False

        count = 0
        for i in range(10**(n-1), 10**n):
            if is_good(i, k):
                count += 1
        return count