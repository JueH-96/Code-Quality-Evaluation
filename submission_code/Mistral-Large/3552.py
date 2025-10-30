class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        def is_palindrome(x):
            return str(x) == str(x)[::-1]

        def generate_palindrome(first_half):
            first_half_str = str(first_half)
            if n % 2 == 0:
                return first_half_str + first_half_str[::-1]
            else:
                return first_half_str + first_half_str[-2::-1]

        upper_bound = 10**n
        for num in range(upper_bound - 1, -1, -1):
            if num % k == 0 and is_palindrome(num):
                return str(num)

        return ""