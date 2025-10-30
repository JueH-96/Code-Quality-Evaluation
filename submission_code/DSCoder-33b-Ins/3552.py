class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        def create_palindrome(n):
            return int(str(n) + str(n)[::-1])

        def is_k_palindrome(n, k):
            return n % k == 0

        upper = 10 ** n - 1
        lower = 10 ** (n - 1) - 1

        for i in range(upper, lower, -1):
            palindrome = create_palindrome(i)
            if is_k_palindrome(palindrome, k):
                return str(palindrome)

        return ''